"""Set geometry: the coherent subspace of a set, and what lies on or off it.

Three reads, one mechanism. A set of embeddings has a coherent subspace — the
directions standing above its own noise floor. Ask how much of a *target* lies inside
it and you have grounding. Ask how much of each *member* lies inside it and you have
anomaly. Ask how much of a *new item* lies outside it and you have novelty. An injected
instruction and a genuinely new source are the same object read with opposite sign:
a mode off the shared structure.

The advantage over a cosine-to-centroid grows with the number of sources the context
spans, and there is none at one: a centroid points one way, and a context assembled from
several documents does not. Measured on real attack strings
(`deepset/prompt-injections`) planted in wikitext-103 articles, AUROC 0.907 against the
centroid's 0.733 at five sources; on the on-topic decision, 0.872 against 0.717. Top-1
localisation leads at every source count, 0.160 against 0.053 at five.
research/PAPER.md has the construction and the conditions.

What this is not
----------------
**Coherence is not factuality.** It reads topic. A wrong-but-on-topic answer reads as
coherent, because staying on a subject and being right about it are different
properties and only the first is visible in the geometry. Use it for off-source and
lost-thread monitoring. Do not pitch it as a factual abstain signal.

**Anomaly is a score and a localisation.** It exposes no binary "does this set contain a
foreign item". From one set alone that is ill-posed —
the most peripheral legitimate item of a clean multi-topic set and a genuinely foreign
item of an easier set carry overlapping absolute anomaly. What is exposed is the
per-item score, the ranking, and the top-1 "inspect this first". A calibrated binary
decision needs a reference distribution of known-clean anomalies, which the caller
supplies; :func:`anomaly_z` is that form.

Each item is scored against the subspace of the set **without it**, because scoring it
against a subspace it helped define inverts the read on exactly the item you were
looking for. :func:`anomaly` carries the measurement. The fast in-set read is still
reachable, and is what the published AUROC was measured with.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import engine
from .engine import CutUnavailable

__all__ = [
    "coherent_basis",
    "in_subspace_fraction",
    "subspace_overlap",
    "chance_overlap",
    "coherence",
    "grounding",
    "anomaly",
    "anomaly_rank",
    "most_anomalous",
    "anomaly_z",
    "novelty_score",
    "assess_coherence",
    "assess_novelty",
]

_UNIT_NORM_FLOOR = 1e-9


def coherent_basis(E):
    """``(V, K)`` — the coherent subspace of a row-set ``E`` ``(n, d)``.

    ``V`` is ``(d, K)`` with orthonormal columns: the top-``K`` right singular
    directions of the direction-normalised set, ``K`` being the engine's resolved-mode
    count. Rows are normalised to unit length first so the subspace is set by where the
    items point, not by which of them happens to have the largest norm.

    The modes are taken off the normalised set as given rather than the whitened frame.
    Whitening subtracts each coordinate's median, and for a set of embeddings the shared
    mean direction *is* the topic — the leading coherent mode — so removing it would
    delete the thing being asked about. The rank still comes from the whitened frame,
    where the noise floor is meaningful; see :mod:`entroptics_llm.engine`.
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2 or min(E.shape) < 2:
        raise CutUnavailable(f"a {E.shape} set has no subspace to read")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + _UNIT_NORM_FLOOR)
    _, _, Vt = np.linalg.svd(En, full_matrices=False)
    K = max(1, min(engine.resolved_rank(En), Vt.shape[0]))
    return Vt[:K].T, K


def in_subspace_fraction(v, V) -> float:
    """``[0, 1]`` — the fraction of ``v``'s energy lying inside subspace ``V``.

    1 = entirely explained by those directions, 0 = entirely off them.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    vn = v / (np.linalg.norm(v) + _UNIT_NORM_FLOOR)
    proj = np.asarray(V, dtype=np.float64).T @ vn
    return float(np.dot(proj, proj))


def subspace_overlap(V_a, V_b) -> float:
    """``[0, 1]`` — how much of subspace ``V_a`` is covered by subspace ``V_b``.

    ``||V_a^T V_b||_F^2 / k_a``: the mean squared principal cosine between the two, a
    subspace-to-subspace generalisation of :func:`in_subspace_fraction`. Use it when
    both sides are sets — one context's sources against another's, or the same context
    before and after a turn.

    Read it against :func:`chance_overlap`, always. In high dimension two subspaces
    that are not specifically aligned are near-orthogonal by default, so a raw overlap
    is uninterpretable and a raw *non*-overlap is uninformative — a read that reports
    "~100% of the structure lies outside" for a random matrix as readily as for a real
    one is measuring the ambient dimension, not the relationship.
    """
    A = np.asarray(V_a, dtype=np.float64)
    B = np.asarray(V_b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
        raise CutUnavailable(f"subspaces must share their ambient dim; got {A.shape}, {B.shape}")
    M = A.T @ B
    return float((M * M).sum() / max(1, A.shape[1]))


def chance_overlap(k_b: int, d: int) -> float:
    """``k_b / d`` — what :func:`subspace_overlap` returns for an unrelated ``V_b``.

    The analytic null: a random ``k_b``-dimensional subspace covers ``k_b/d`` of
    anything. Report the ratio of measured overlap to this, not the overlap alone.
    """
    return float(k_b) / float(max(1, d))


def coherence(target_emb, context_embs) -> float:
    """``[0, 1]`` — how much of ``target_emb`` lies in the context's coherent subspace.

    Higher is more on-topic; a sharp drop flags drift. Topic coherence only — see the
    module docstring.
    """
    V, _ = coherent_basis(context_embs)
    return in_subspace_fraction(target_emb, V)


#: The same read under the name the verification use-case calls it. One implementation.
grounding = coherence


def anomaly(cand_embs, *, leave_one_out: bool = True) -> np.ndarray:
    """``(n,)`` in ``[0, 1]`` — per item, the energy *outside* the set's coherent subspace.

    High = a foreign mode off the shared structure.

    Each item is scored against the subspace of the set **without it**. That is the
    default because the alternative inverts, and not rarely.

    Why the in-set read inverts
    ---------------------------
    Reading an item against a subspace **the item itself helped define** means an item
    foreign enough to clear the noise floor on its own becomes its own coherent mode and
    reads as maximally *normal*. The read does not weaken; it flips. Measured, one
    injected item in a clean multi-topic set, 15 draws, top-1 localisation:

        injected item is        1 topic   3 topics   5 topics
        70% off the topics       1.00       1.00       1.00      (both reads)
        90% off                  1.00       0.80       1.00      in-set
                                 1.00       1.00       1.00      leave-one-out
        fully orthogonal         1.00       0.07       1.00      in-set
                                 1.00       1.00       1.00      leave-one-out

    Note it is **not monotone in foreignness or in topic count** — it depends on whether
    the resolved rank happens to reach the outlier's own mode, which a caller cannot
    predict from anything they can see. Leave-one-out scored 1.00 in every cell, so it is
    not a trade: it never lost. ``research/PAPER.md`` appendix A carries this table.

    ``leave_one_out=False`` is the fast read — real injections are text, they land inside
    the embedder's cone, and a partial outlier does not earn a mode. Pass it when latency
    rules and the set is known not to contain a hard outlier.

    Cost: ``n`` decompositions instead of one. Measured — n=20 d=384: 3.3 ms vs 59 ms
    (18x); n=40 d=384: 8.4 ms vs 287 ms (34x); and the gap widens with n.
    """
    E = np.asarray(cand_embs, dtype=np.float64)
    if not leave_one_out:
        V, _ = coherent_basis(E)
        return np.array([1.0 - in_subspace_fraction(E[i], V) for i in range(len(E))])
    if len(E) < 3:
        raise CutUnavailable(
            f"leave-one-out needs at least 3 items to leave one out of; got {len(E)}"
        )
    out = np.empty(len(E))
    for i in range(len(E)):
        V, _ = coherent_basis(np.delete(E, i, axis=0))
        out[i] = 1.0 - in_subspace_fraction(E[i], V)
    return out


def anomaly_rank(cand_embs, *, leave_one_out: bool = True) -> np.ndarray:
    """Item indices, most anomalous first. ``anomaly_rank(E)[0]`` is the most foreign."""
    return np.argsort(anomaly(cand_embs, leave_one_out=leave_one_out))[::-1]


def most_anomalous(cand_embs, *, leave_one_out: bool = True):
    """The index of the single most anomalous item, or ``None`` for an empty set.

    Top-1 localisation — which item to inspect first. Not a verdict that the set
    contains a foreign item; see the module docstring for why that question is not well
    posed from one set, and :func:`anomaly` for what ``leave_one_out=False`` costs.
    """
    E = np.asarray(cand_embs, dtype=np.float64)
    if len(E) == 0:
        return None
    if leave_one_out and len(E) < 3:
        leave_one_out = False
    return int(np.argmax(anomaly(E, leave_one_out=leave_one_out)))


def anomaly_z(cand_embs, reference_anomalies, *, leave_one_out: bool = True) -> np.ndarray:
    """``(n,)`` — per-item anomaly in standard deviations above a reference distribution.

    ``reference_anomalies`` is a sample of :func:`anomaly` values taken over sets known
    to be clean, from the same corpus and embedder. This is the only honest route to a
    binary decision: the caller brings the reference, sets the cut, and owns the
    false-alarm rate that follows.

    Raises rather than inventing a scale when the reference has none — a reference of
    one value, or of identical values, cannot standardise anything, and returning
    zeros or infinities would read as a confident answer.
    """
    ref = np.asarray(reference_anomalies, dtype=np.float64).ravel()
    if ref.size < 2:
        raise CutUnavailable(
            f"a reference of {ref.size} value(s) has no spread to standardise against"
        )
    sd = float(ref.std(ddof=1))
    if sd <= 0.0:
        raise CutUnavailable(
            "the reference anomalies are identical; there is no scale to report z against"
        )
    return (anomaly(cand_embs, leave_one_out=leave_one_out) - float(ref.mean())) / sd


def novelty_score(new_emb, corpus_embs) -> float:
    """``[0, 1]`` — how much of ``new_emb`` lies off the corpus's coherent subspace.

    1 = entirely foreign, and so genuinely new; 0 = explained by the modes already
    there, and so corroboration rather than information.
    """
    V, _ = coherent_basis(corpus_embs)
    return float(1.0 - in_subspace_fraction(new_emb, V))


def assess_coherence(target_emb, context_embs, min_context: int = 4) -> dict[str, Any]:
    """:func:`coherence` with its evidence, and never raising.

    Returns numbers only — turning a score into a label (grounded / drifted) is a
    display concern and is left to the consumer, because the cut point belongs to
    whoever owns the consequence of being wrong.

    ``available`` False means the read did not happen, and the caller must do something
    other than treat ``grounding`` as 0.
    """
    info: dict[str, Any] = {
        "available": False, "grounding": None, "context_size": 0, "subspace_dim": 0,
    }
    ctx = np.asarray(context_embs, dtype=np.float64)
    info["context_size"] = len(ctx)
    if len(ctx) < min_context:
        return info
    try:
        V, K = coherent_basis(ctx)
        score = in_subspace_fraction(target_emb, V)
    except CutUnavailable:
        return info
    info.update({"available": True, "grounding": round(float(score), 4), "subspace_dim": int(K)})
    return info


def assess_novelty(new_emb, corpus_embs, min_corpus: int = 4) -> dict[str, Any]:
    """:func:`novelty_score` with its evidence, and never raising. See :func:`assess_coherence`."""
    info: dict[str, Any] = {
        "available": False, "novelty_score": None, "corpus_size": 0, "subspace_dim": 0,
    }
    corpus = np.asarray(corpus_embs, dtype=np.float64)
    info["corpus_size"] = len(corpus)
    if len(corpus) < min_corpus:
        return info
    try:
        V, K = coherent_basis(corpus)
        score = 1.0 - in_subspace_fraction(new_emb, V)
    except CutUnavailable:
        return info
    info.update({"available": True, "novelty_score": round(float(score), 4), "subspace_dim": int(K)})
    return info
