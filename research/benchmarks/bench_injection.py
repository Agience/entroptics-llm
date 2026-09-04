"""Locating an injected instruction inside a multi-source context.

The read
--------
An injected instruction is a **foreign mode**: it does not lie in the coherent subspace the
legitimate content spans. `subspace.anomaly` scores each item by the energy it carries
*outside* that subspace, so the injection stands out for a structural reason rather than a
lexical one — no keyword list, no classifier, no threshold, and no training.

The baseline is the same question asked with one vector instead of a subspace: anomaly as
distance from the centroid of the context.

Why the topic count is the axis
-------------------------------
A centroid is a single direction, so it can only represent a context that points one way.
Give it a context drawn from several sources and the centroid sits between them, in a
direction no real item occupies — and legitimate-but-peripheral items then score as more
foreign than the injection does. A subspace spans the sources instead, so the legitimate
material lies *in* it however many topics it covers, and only the injection lies off it.

That is why this is measured against the number of sources in the context rather than at one
setting. A retrieval-augmented context spanning several documents is the ordinary case, not a
stress test.

The data is real on both sides: legitimate content is wikitext-103 articles, and the attacks
are the `deepset/prompt-injections` corpus rather than strings written for the occasion.

Run::

    python -m research.benchmarks.bench_injection
    python -m research.benchmarks.bench_injection --topics 1,2,3,5,8 --trials 300
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from entroptics_llm import anomaly, coherent_basis
from entroptics_llm.engine import CutUnavailable

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def articles(min_sentences: int, wanted: int, max_sentences: int = 40):
    """Multi-section wikitext-103 articles, as sentence lists.

    A whole article is one topic. Splitting on the top-level ``= Title =`` header keeps each
    one intact, so a context assembled from M articles genuinely spans M subjects rather than
    M slices of the same subject.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    header = re.compile(r"^ = [^=].* = $")
    arts, cur, out = [], [], []
    for row in ds:
        line = row["text"].rstrip("\n")
        if header.match(line):
            if cur:
                arts.append(" ".join(cur))
                cur = []
        elif line.strip():
            cur.append(line.strip())
        if len(arts) >= wanted * 4:
            break
    for a in arts:
        s = [x for x in _SENTENCE.split(a) if len(x.split()) >= 5]
        if len(s) >= min_sentences:
            out.append(s[:max_sentences])
        if len(out) >= wanted:
            break
    return out


def attacks(limit: int):
    """Real prompt injections. Label 1 is an attack in this corpus."""
    from datasets import load_dataset

    ds = load_dataset("deepset/prompt-injections", split="train")
    return [r["text"] for r in ds if int(r["label"]) == 1][:limit]


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUROC by rank with ties averaged. Local so the benchmark needs no sklearn."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def boot(d: np.ndarray, n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    return d.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--topics", default="1,2,3,5")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--min-sentences", type=int, default=6)
    ap.add_argument("--knn", type=int, default=5,
                    help="neighbours for the k-NN outlier baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    Ms = [int(x) for x in args.topics.split(",")]

    from sentence_transformers import SentenceTransformer

    arts = articles(args.min_sentences, max(80, max(Ms) * 30))
    atk = attacks(200)
    model = SentenceTransformer(args.encoder, device="cpu")
    model.eval()
    print(f"encoder {args.encoder} | {len(arts)} articles | {len(atk)} real injections "
          f"| {args.trials} trials per topic count", flush=True)

    embs = [np.asarray(model.encode(s, normalize_embeddings=True, show_progress_bar=False),
                       dtype=np.float64) for s in arts]
    inj = np.asarray(model.encode(atk, normalize_embeddings=True, show_progress_bar=False),
                     dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    print("\n" + "=" * 78)
    print("LOCATING A REAL PROMPT INJECTION IN A CONTEXT SPANNING M SOURCES")
    print("=" * 78)
    print(f"\n{'sources':>8} {'modes':>6} {'items':>6} | {'entroptics':>10} {'centroid':>9} {'k-NN':>7} "
          f"{'vs centroid [95% CI]':>23} {'vs k-NN [95% CI]':>23} | {'top-1 ent':>10} {'top-1 cen':>10} {'top-1 knn':>10}")
    print("-" * 78)
    rows = []
    for M in Ms:
        sub_a, cen_a, knn_a, sub_t, cen_t, knn_t, Ks, sizes = ([] for _ in range(8))
        for _ in range(args.trials):
            idx = rng.choice(len(arts), size=M, replace=False)
            legit = np.vstack([embs[a] for a in idx])
            one = inj[rng.integers(len(inj))][None, :]
            E = np.vstack([legit, one])          # the injection's position is not known to the read
            label = np.r_[np.zeros(len(legit), bool), np.ones(1, bool)]
            try:
                V, K = coherent_basis(E)
            except CutUnavailable:
                continue
            centroid = E.mean(0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            s_sub = anomaly(E)          # leave-one-out, the library's default
            s_cen = 1.0 - (E @ centroid)
            # k-NN distance: the other standard unsupervised outlier score, and one that is
            # NOT a monotone transform of the centroid read. An item is scored by how far it
            # sits from its k nearest neighbours, so a foreign item in a set with several
            # dense clusters is judged against the cluster it is nearest to rather than
            # against the whole set's average direction.
            S = E @ E.T
            np.fill_diagonal(S, -np.inf)
            k = min(args.knn, len(E) - 1)
            s_knn = 1.0 - np.sort(S, axis=1)[:, -k:].mean(1)
            sub_a.append(auroc(s_sub, label))
            cen_a.append(auroc(s_cen, label))
            knn_a.append(auroc(s_knn, label))
            knn_t.append(int(s_knn.argmax() == len(E) - 1))
            sub_t.append(int(s_sub.argmax() == len(E) - 1))
            cen_t.append(int(s_cen.argmax() == len(E) - 1))
            Ks.append(K)
            sizes.append(len(E))
        d = np.array(sub_a) - np.array(cen_a)
        mm, lo, hi = boot(d)
        # The comparison the claim stands on: entroptics against the strongest standard
        # baseline, paired per trial so the two saw the same context and the same attack.
        km, klo, khi = boot(np.array(sub_a) - np.array(knn_a))
        rows.append({"M": M, "modes": float(np.mean(Ks)), "n": len(sub_a),
                     "subspace": float(np.mean(sub_a)), "centroid": float(np.mean(cen_a)),
                     "knn": float(np.mean(knn_a)), "top1_knn": float(np.mean(knn_t)),
                     "edge": mm, "ci": [lo, hi],
                     "edge_vs_knn": km, "ci_vs_knn": [klo, khi],
                     "top1_subspace": float(np.mean(sub_t)),
                     "top1_centroid": float(np.mean(cen_t))})
        print(f"{M:>8} {np.mean(Ks):>6.1f} {np.mean(sizes):>6.0f} | "
              f"{np.mean(sub_a):>10.3f} {np.mean(cen_a):>9.3f} {np.mean(knn_a):>7.3f} "
              f"{mm:>+8.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{km:>+8.3f} [{klo:+.3f},{khi:+.3f}] | "
              f"{np.mean(sub_t):>10.3f} {np.mean(cen_t):>10.3f} {np.mean(knn_t):>10.3f}", flush=True)
    print("-" * 78)
    if len(rows) > 1:
        print(f"edge at {rows[0]['M']} source(s): {rows[0]['edge']:+.3f}   "
              f"at {rows[-1]['M']}: {rows[-1]['edge']:+.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"encoder": args.encoder, "trials": args.trials, "rows": rows}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
