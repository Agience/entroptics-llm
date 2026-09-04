"""The retrieval cut on real corpora: BEIR, real embeddings, real relevance judgements.

Why this file exists
--------------------
Every real-corpus retrieval number this replaced was produced by the
**pre-repair lock** and was never re-run — the BeIR scifact and fiqa figures, the HotpotQA
distractor result, the LongMemEval bake-off. This measures the shipped lock on real
corpora with human judgements, which is the question a planted-facet sweep leaves open.

This does. Nothing here is planted and nothing is fitted.

What is compared, and why it is the fair comparison
---------------------------------------------------
The cut is a **reducer over a horizon**, not a retriever — the package's own scope note
says so, and LongMemEval is the measured reason (0.28 recall against top-8's 0.74 when
used *as* the retriever). So the horizon is fixed first, by an ordinary dense retriever,
and every method below decides the same question: *of these B candidates, which do we
keep?*

    fixed-k        top-1 / top-3 / top-5 / top-10 out of the horizon
    cosine+lock    the SAME parameter-free lock, applied to cosine salience
    cut            the query-relative multi-head screen, then that lock

``cosine+lock`` is the baseline that matters and the one a published +0.038 edge was
once wrong about: that gap was the *lock*, not the screen, because the two sides were
running different locks. Here both sides run
:func:`~entroptics_llm.lock.top_break` on a non-negative salience, and the only
difference between them is the screen.

Negative cosines are clipped at 0 for the baseline's salience, which is exactly what
``head_screen`` does to its own per-head cosines. The lock refuses a negative spectrum —
a multiplicative gap between negative values is not a drop — so *some* map to the
non-negative half-line is forced, and matching the screen's own is the choice that leaves
the screen as the sole difference.

Two F1s are reported and they answer different questions
--------------------------------------------------------
``F1@horizon``  — against the relevant documents *that the horizon contains*. This scores
                  the selection decision alone, and is the number the cut is about.
``F1@corpus``   — against the full judged relevant set. This includes the retriever's own
                  misses, so every method here shares the same ceiling; it is reported so
                  the selection edge is not read as an end-to-end claim.

Run::

    python -m research.benchmarks.bench_beir                       # nfcorpus + scifact
    python -m research.benchmarks.bench_beir --datasets fiqa arguana
    python -m research.benchmarks.bench_beir --model BAAI/bge-small-en-v1.5 --horizon 50

Determinism: the encoder is in eval mode with no sampling, the lock carries no RNG, and
the only randomness is the bootstrap, which takes ``--seed``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

from entroptics_llm import select_with_info
from entroptics_llm.lock import top_break

#: `scidocs` is here because the head/tail claim needed a corpus whose queries are
#: ACTUALLY multi-document: nearly every one of its 1,000 queries carries four or more relevant
#: papers, against nfcorpus's minority tail and essentially none in scifact or
#: arguana. A claim about wide-evidence queries tested only where wide evidence is
#: rare is a claim tested on a handful of rows.
DATASETS = ("nfcorpus", "scifact", "fiqa", "arguana", "scidocs")
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: A second encoder is not decoration. The headline claim is that each corpus has a
#: different best fixed `k`, and its own stated falsification is a mixture whose optima all
#: agree -- so "the optima only differ because of THIS embedding model" is the first
#: objection anyone will raise. `--model BAAI/bge-small-en-v1.5` re-runs the whole
#: comparison under a stronger encoder of the same width.
ALT_MODEL = "BAAI/bge-small-en-v1.5"


# ── corpus ────────────────────────────────────────────────────────────────────

def load_beir(name: str, split: str = "test"):
    """``(doc_ids, doc_texts, query_ids, query_texts, relevant)`` from the HF mirror.

    ``relevant`` maps a query id to the set of doc ids judged relevant. BEIR qrels carry
    a graded ``score``; anything ``>= 1`` is relevant, which is the standard binarisation
    and the one the published scifact/fiqa numbers used.

    Ids are normalised to ``str`` because the mirrors are not consistent about it —
    nfcorpus stores them as strings and scifact as integers, and joining the two forms
    produces an empty relevant set rather than an error.
    """
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split=split)

    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row["score"]) >= 1:
            relevant[str(row["query-id"])].add(str(row["corpus-id"]))

    doc_ids = [str(x) for x in corpus["_id"]]
    titles, texts = corpus["title"], corpus["text"]
    doc_texts = [(t + " " + b).strip() if t else b for t, b in zip(titles, texts)]

    q_ids, q_texts = [], []
    for qid, qtext in zip(queries["_id"], queries["text"]):
        qid = str(qid)
        if qid in relevant:
            q_ids.append(qid)
            q_texts.append(qtext)
    return doc_ids, doc_texts, q_ids, q_texts, relevant


def encode(model, texts, batch_size: int, label: str, cache_dir=None, key: str = ""):
    """Encode, reusing a cached matrix when one matches.

    Encoding is the whole cost of this benchmark — 160 s for nfcorpus's 3,633 documents
    on CPU, 194 s for scifact's 5,183 — and it is the same every run, so a re-run to
    change a *selection* rule should not pay it again. The cache key carries the model
    name and the text count; a mismatch re-encodes rather than returning the wrong
    matrix.
    """
    import hashlib

    path = None
    if cache_dir and key:
        stamp = hashlib.sha256(
            (key + "|" + str(len(texts)) + "|" + (texts[0] if texts else "")).encode("utf-8")
        ).hexdigest()[:16]
        path = pathlib.Path(cache_dir) / f"{key}-{stamp}.npy"
        if path.exists():
            E = np.load(path)
            if E.shape[0] == len(texts):
                print(f"    cached  {label:8s} {len(texts):6d}", flush=True)
                return E
    t0 = time.time()
    E = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    E = np.asarray(E, dtype=np.float32)
    print(f"    encoded {label:8s} {len(texts):6d} in {time.time() - t0:6.1f}s", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, E)
    return E


# ── the methods ───────────────────────────────────────────────────────────────

def f1(kept: set[str], gold: set[str]) -> float:
    """F1 of a selected set against a judged set. Empty selection scores 0."""
    if not kept or not gold:
        return 0.0
    tp = len(kept & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(kept), tp / len(gold)
    return 2 * p * r / (p + r)


def cosine_lock(sims: np.ndarray) -> np.ndarray:
    """The fair baseline: the shipped lock on cosine salience, clipped as the screen clips."""
    keep, _ = top_break(np.clip(sims, 0.0, None))
    return np.where(keep)[0]


def run_dataset(name, model, *, horizon, split, batch_size, min_horizon, cache_dir=None):
    print(f"  {name}: loading", flush=True)
    doc_ids, doc_texts, q_ids, q_texts, relevant = load_beir(name, split)
    print(f"    {len(doc_texts)} docs, {len(q_ids)} judged queries", flush=True)

    D = encode(model, doc_texts, batch_size, "corpus", cache_dir, f"{name}-corpus")
    Q = encode(model, q_texts, batch_size, "queries", cache_dir, f"{name}-queries-{split}")

    doc_index = {d: i for i, d in enumerate(doc_ids)}
    methods = ["top1", "top3", "top5", "top10", "cosine+lock", "cut"]
    per_query = {m: {"h": [], "c": []} for m in methods}
    kept_n = {m: [] for m in methods}
    horizon_recall, fell_back, found_gap, heads = [], 0, 0, []

    t0 = time.time()
    for i, qid in enumerate(q_ids):
        sims = D @ Q[i]
        # A query may BE a corpus document — arguana's queries are arguments and the
        # gold is the counterargument, so the nearest neighbour of every query is its own
        # text. Left in, every method spends its budget on the self-match and the table
        # measures that artifact instead of the selection: measured, top-1 scored F1
        # 0.000 on arguana and both locks kept ~1.4 documents at F1 0.02. Excluded here
        # for every dataset, which is a no-op where query ids and document ids are
        # disjoint.
        self_match = doc_index.get(qid)
        if self_match is not None:
            sims[self_match] = -np.inf
        B = min(horizon, len(sims))
        top = np.argpartition(-sims, B - 1)[:B]
        top = top[np.argsort(-sims[top])]

        gold = relevant[qid]
        pool_ids = [doc_ids[j] for j in top]
        in_pool = set(pool_ids) & gold
        horizon_recall.append(len(in_pool) / len(gold))
        if not in_pool:
            continue                      # nothing to select; scores 0 for every method alike

        pool_sims = sims[top]
        picks = {
            "top1": np.arange(min(1, B)),
            "top3": np.arange(min(3, B)),
            "top5": np.arange(min(5, B)),
            "top10": np.arange(min(10, B)),
            "cosine+lock": cosine_lock(pool_sims),
        }
        idx, info = select_with_info(D[top], Q[i], min_horizon=min_horizon)
        picks["cut"] = np.asarray(idx, dtype=int)
        fell_back += bool(info["fell_back"])
        found_gap += bool(info["found_gap"])
        heads.append(info["heads_derived"])

        for m, sel in picks.items():
            chosen = {pool_ids[j] for j in sel}
            per_query[m]["h"].append(f1(chosen, in_pool))
            per_query[m]["c"].append(f1(chosen, gold))
            kept_n[m].append(len(sel))

    elapsed = time.time() - t0
    n_scored = len(per_query["cut"]["h"])
    return {
        "dataset": name,
        "n_docs": len(doc_texts),
        "n_queries": len(q_ids),
        "n_scored": n_scored,
        "horizon": horizon,
        "horizon_recall": float(np.mean(horizon_recall)) if horizon_recall else 0.0,
        "heads_median": float(np.median(heads)) if heads else 0.0,
        "fell_back": fell_back,
        "found_gap": found_gap,
        "seconds": elapsed,
        "f1_horizon": {m: float(np.mean(v["h"])) for m, v in per_query.items()},
        "f1_corpus": {m: float(np.mean(v["c"])) for m, v in per_query.items()},
        "kept_mean": {m: float(np.mean(v)) for m, v in kept_n.items()},
        "_paired": {m: np.asarray(v["h"], dtype=np.float64) for m, v in per_query.items()},
    }


def bootstrap_edge(a: np.ndarray, b: np.ndarray, *, n: int = 10000, seed: int = 0):
    """Paired bootstrap 95% CI on ``mean(a) - mean(b)``. Paired, because both methods
    answered the same queries and an unpaired interval would be wider for no reason."""
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── report ────────────────────────────────────────────────────────────────────

def report(results, *, model_name, seed):
    methods = ["top1", "top3", "top5", "top10", "cosine+lock", "cut"]
    print()
    print("=" * 100)
    print(f"THE RETRIEVAL CUT ON BEIR — real corpora, real judgements. encoder: {model_name}")
    print("=" * 100)
    print("\nF1 against the relevant documents the horizon contains (the selection decision):\n")
    head = f"{'dataset':10s} {'n':>5s} {'hzn-rec':>8s} " + " ".join(f"{m:>12s}" for m in methods)
    print(head)
    print("-" * len(head))
    for r in results:
        row = f"{r['dataset']:10s} {r['n_scored']:5d} {r['horizon_recall']:8.3f} "
        row += " ".join(f"{r['f1_horizon'][m]:12.3f}" for m in methods)
        print(row)

    print("\nF1 against the full judged set (shares the retriever's ceiling on every row):\n")
    print(head)
    print("-" * len(head))
    for r in results:
        row = f"{r['dataset']:10s} {r['n_scored']:5d} {r['horizon_recall']:8.3f} "
        row += " ".join(f"{r['f1_corpus'][m]:12.3f}" for m in methods)
        print(row)

    print("\nMean number kept (the cut's is derived; every other column is typed in):\n")
    print(head)
    print("-" * len(head))
    for r in results:
        row = f"{r['dataset']:10s} {r['n_scored']:5d} {'':>8s} "
        row += " ".join(f"{r['kept_mean'][m]:12.2f}" for m in methods)
        print(row)

    print("\nThe edge over the fair baseline, paired bootstrap 95% CI (F1@horizon):\n")
    print(f"{'dataset':10s} {'entroptics':>11s} {'cos+lock':>10s} {'edge':>9s} {'95% CI':>20s}  verdict")
    print("-" * 74)
    for r in results:
        p = r["_paired"]
        e, lo, hi = bootstrap_edge(p["cut"], p["cosine+lock"], seed=seed)
        v = "cut wins" if lo > 0 else ("baseline wins" if hi < 0 else "tie — CI spans 0")
        print(f"{r['dataset']:10s} {r['f1_horizon']['cut']:11.3f} "
              f"{r['f1_horizon']['cosine+lock']:10.3f} {e:+9.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]".ljust(56) + f"  {v}")

    print("\nEvidence the reads carried:\n")
    print(f"{'dataset':10s} {'heads':>7s} {'found_gap':>11s} {'fell_back':>11s} {'seconds':>9s}")
    print("-" * 52)
    for r in results:
        print(f"{r['dataset']:10s} {r['heads_median']:7.0f} "
              f"{r['found_gap']:11d} {r['fell_back']:11d} {r['seconds']:9.1f}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact"], choices=DATASETS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--horizon", type=int, default=50,
                    help="candidate pool size from the dense retriever (default 50)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-horizon", type=int, default=8)
    ap.add_argument("--cache-dir", default=None,
                    help="reuse encoded corpora across runs; encoding is the whole cost")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed only")
    ap.add_argument("--json", default=None, help="also write the full result to this path")
    args = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer
    print(f"loading encoder {args.model}", flush=True)
    model = SentenceTransformer(args.model, device="cpu")
    model.eval()

    results = []
    for name in args.datasets:
        try:
            results.append(run_dataset(name, model, horizon=args.horizon, split=args.split,
                                       batch_size=args.batch_size, min_horizon=args.min_horizon,
                                       cache_dir=args.cache_dir))
        except Exception as e:                                   # noqa: BLE001 — report, do not abort the sweep
            print(f"  {name}: FAILED — {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    if not results:
        print("no dataset completed", file=sys.stderr)
        return 1

    report(results, model_name=args.model, seed=args.seed)

    if args.json:
        out = []
        for r in results:
            r = {k: v for k, v in r.items() if k != "_paired"}
            out.append(r)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "results": out}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
