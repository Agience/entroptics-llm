"""The retrieval cut: which of a candidate pool to keep for a query.

The difficulty this solves
--------------------------
The coherent modes of a raw candidate set denote *intrinsic* structure — the dominant
shared directions, equivalently the outliers. Retrieval relevance is *extrinsic*: it
is defined by the query, and a relevant item sits near the query rather than away from
the crowd. Reading the candidate embeddings directly therefore selects the atypical
items and never the gold; measured F1 **0.000**. Two other obvious routes fail too: an
entropy-scale cut on the score distribution finds it too uniform to fold and correctly
refuses, and word- or sub-query facets are cruder than the full-query cosine.

What works is to make the screen query-relative before reading it. Subdivide each
embedding into ``H`` contiguous heads, build ``W[item, head]`` = the per-head cosine of
that item's block to the query's block, and read *that*. Now the screen's coherent
structure means "matches the query across the heads" — a gold item — rather than "is
an outlier", and the intrinsic read becomes an extrinsic one without changing the
engine.

What it is measured to be worth
-------------------------------
On four real BEIR corpora, against the identical ``top_break`` lock run on plain cosine
salience so the screen is the only difference, this ties on all four, every interval
containing zero. That is the whole of the real-corpus evidence, and it does not include a
win or a loss.

The reason they tie is the law this package is built on: BEIR relevance is
unimodal — one claim wants one supporting abstract, one argument wants one
counterargument — and at one coherent mode this read **is** cosine. A tie there is the
correct result rather than a shortfall. The screen's stated win condition is
multi-modal relevance, and it has not been demonstrated on a real corpus.

What the module *is* measured to buy is the derived count rather than an edge over
cosine: the best single constant *k* gives up as much as 0.086 F1 against a corpus's
own tuned *k*, where reading the count per query gives up at most 0.019. That is a
bound on the cost of having no constant to sweep, not a claim of superiority.

It is precision-oriented and does not do bulk recall: the lock isolates one clean cluster
above a gap, so it under-selects when recall is what is wanted. That has a measured price,
and the price is the COUNT rather than the ranking. End to end on HotpotQA, 150 questions:
handed the same number of paragraphs it derived, cosine scores the same answers (+0.000 at
1.5B, -0.001 at 7B), so the selection is not the problem. Against a fixed top-5 it spends
36% of the context tokens and costs 0.133 F1 with Qwen2.5-7B-Instruct, all of it because it
keeps 2.1 paragraphs where a multi-hop answer needs two. Use a generous top-k, put this read
downstream of it, and do not use it where the evidence spans passages.

This is a separate and weaker result than the claim in ``research/PAPER.md``, and it is
separate from it. research/PAPER.md section 7 carries every row.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import engine
from .engine import CutUnavailable
from .lock import top_break

__all__ = [
    "cosine_salience", "select_by_cosine",
    "head_screen",
    "signal_power",
    "derive_heads",
    "select",
    "select_with_info",
    "MIN_HEADS",
]

#: Added to every norm before dividing by it. Stated, not derived — a guard against
#: dividing by a zero-length vector. The honest replacement is the forward error of the
#: summation actually performed, read off the frame's own dtype; it is not applied
#: because ``|v| + 1e-9`` and ``|v| + tiny`` are different denominators and every
#: downstream number would move.
_UNIT_NORM_FLOOR = 1e-9


def _smallest_readable_head_count() -> int:
    """The fewest heads at which :func:`signal_power`'s own read can be taken.

    Recovered from the engine's gate rather than typed. ``head_screen`` hands
    ``signal_power`` an ``(n, heads)`` matrix, which asks the engine for its resolved
    rank; the engine gives no reading below ``min(n, heads) >= 2``. A one-head screen
    is therefore a screen whose read cannot be taken, while ``signal_power`` would
    still return a number — a single column's energy dressed as a spectral read.

    The probe holds the row axis one wider than the head axis so the heads are always
    the binding side, and uses distinct column values so nothing but the shape can make
    it unreadable. ``min(h + 1, h) = h`` is strictly increasing, so this terminates by
    the algebra and needs no iteration bound. It lands on 2.
    """
    heads = 0
    while True:
        heads += 1
        probe = np.arange(1.0, float((heads + 1) * heads) + 1.0).reshape(heads + 1, heads)
        try:
            engine.resolved_rank(probe)
        except CutUnavailable:
            continue
        return heads


#: The floor on the derived head count — the engine's readability gate seen from the
#: head axis. Computed, and equal to 2.
MIN_HEADS = _smallest_readable_head_count()

#: The ceiling on the derived head count. Two bounds are taken together and the tighter
#: wins. ``n_features // 2`` **is** derived — it is the geometry, since a head block
#: narrower than two coordinates has a "cosine" carrying only a sign. ``64`` is
#: **stated, not derived**, and it is the one that binds.
#:
#: It binds sooner than one might hope, and on structured input too: measured on
#: retrieval-shaped pools the raw count is 18 at n=20, 37 at n=40 and **70 at n=80**, so
#: 64 binds from a horizon of about 80 upward — not only on isotropic sets.
#:
#: It does not decide the answer. Swept across a 12x range of ceilings on the same pools,
#: F1 is flat across hi = 16 / 32 / 64 / 128 / 192. So it is a guard, not a tuning knob
#: wearing a guard's name.
MAX_HEADS = 64


def head_screen(item_embs, query_emb, heads: int) -> np.ndarray:
    """``(n, heads)`` — each item's per-head cosine to the query, block by block.

    The feature axis is cut into ``heads`` contiguous blocks and each block compared on
    its own, so an item matching the query strongly on part of the coordinate and not
    at all elsewhere is visible as such rather than averaged into one scalar.

    Negative cosines are clipped to zero. A block pointing away from the query is not
    negative evidence, it is *no* evidence, and letting it subtract would let two
    irrelevant blocks cancel into an apparent match.

    Consequence of the clip, measured and bounded. For unrelated blocks the cosine is
    symmetric about zero, so the clip makes a large share of every column exactly zero
    and the per-column MAD can be exactly 0 — and that is the frame :func:`signal_power`
    reads its mode count off. Its reach is bounded:

    * On **retrieval-shaped** pools only a minority of each column is zeroed, and breaking
      the degeneracy with a jitter far below the smallest nonzero entry leaves the rank
      unchanged in almost every draw and the F1 essentially unchanged.
    * On **pure-random** pools — no planted relevance, nothing to find — about half of
      each column is zeroed, the rank moves in half the draws, and |dk| reaches 2.

    So the degeneracy bites where there is no signal to read and not where there is.

    The obvious repair is measurably wrong. ``entroptics.occupied_modes`` — the rank
    edge read off the profile's own step, which needs no noise bulk — returns **k=1** on
    a head screen, because the largest step in that spectrum sits right after the
    dominant "everything matches the query a bit" mode rather than at the signal/noise
    boundary. At k=1 the cut *is* cosine, which is the one thing this screen exists to
    stop being. The whitened count stays.
    """
    E = np.asarray(item_embs, dtype=np.float64)
    q = np.asarray(query_emb, dtype=np.float64).ravel()
    if E.ndim != 2:
        raise CutUnavailable(f"item embeddings must be (n, d); got shape {E.shape}")
    if q.shape[0] != E.shape[1]:
        raise CutUnavailable(
            f"query dim {q.shape[0]} != item dim {E.shape[1]} — the blocks would not correspond"
        )
    if not np.isfinite(E).all() or not np.isfinite(q).all():
        raise CutUnavailable("the pool or the query carries NaN or infinite coordinates")
    if np.linalg.norm(q) <= 0.0:
        raise CutUnavailable(
            "the query has zero norm, so it points nowhere and every per-head cosine is 0. "
            "The screen would be all zeros and the lock would keep everything — which is a "
            "read that never happened wearing the shape of one that found no break."
        )
    d = E.shape[1] // heads
    if d < 1:
        raise CutUnavailable(
            f"{heads} heads over {E.shape[1]} features gives an empty block. The screen would "
            f"be all zeros and the lock would keep one arbitrary item while reporting a cut."
        )
    Eb = E[:, : heads * d].reshape(len(E), heads, d)
    qb = q[: heads * d].reshape(heads, d)
    Eb = Eb / (np.linalg.norm(Eb, axis=2, keepdims=True) + _UNIT_NORM_FLOOR)
    qb = qb / (np.linalg.norm(qb, axis=1, keepdims=True) + _UNIT_NORM_FLOOR)
    return np.clip(np.einsum("bhd,hd->bh", Eb, qb), 0.0, None)


def signal_power(W) -> np.ndarray:
    """``(n,)`` — each item's energy in the screen's signal modes.

    Delegates to :func:`entroptics_llm.engine.raw_signal_power`; see that docstring for
    why this substrate keeps its raw magnitudes while the KV substrate does not.
    """
    return engine.raw_signal_power(W)


def derive_heads(item_embs, lo: int | None = None, hi: int | None = None) -> int:
    """The head count, from the candidate set's own effective rank: ``H = round(occupancy * n)``.

    ``occupancy`` is ``2^{H_sv}/n``, the fraction of modes carrying the energy, so
    ``occupancy * n = 2^{H_sv}`` is the effective number of modes the set actually
    occupies. The screen is cut into that many blocks: as many heads as the data has
    independent things to say. Nothing is picked.

    On LongMemEval this self-selects to about 16 per query, which is why a fixed 16 had
    looked like the best choice before it was derived.

    ``lo`` and ``hi`` are not the same kind of number and are documented separately at
    :data:`MIN_HEADS` and :data:`MAX_HEADS`. Passing either overrides the guard, not
    the derivation.

    Rows are normalised to unit length first. Without it the count is read off a
    spectrum dominated by whichever items happen to have the largest norm, and the head
    count then tracks embedding magnitude rather than the set's structure.
    """
    lo = MIN_HEADS if lo is None else int(lo)
    E = np.asarray(item_embs, dtype=np.float64)
    if E.ndim != 2 or min(E.shape) < 2:
        return lo
    ceiling = MAX_HEADS if hi is None else int(hi)
    ceiling = max(lo, min(ceiling, int(E.shape[1]) // 2))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + _UNIT_NORM_FLOOR)
    occ = engine.occupancy(En)
    if not np.isfinite(occ):                 # a set carrying no power has no active modes
        return lo
    return int(max(lo, min(ceiling, round(occ * E.shape[0]))))


def select(item_embs, query_emb, heads: int | None = None) -> np.ndarray:
    """The adaptive cut: which of ``item_embs`` to keep for ``query_emb``, as indices.

    ``heads`` is derived from the data when ``None``; pass an int only to override.
    The lock is :func:`~entroptics_llm.lock.top_break` on the per-item signal power —
    the top cluster above the largest relative gap — so there is no fixed k, no MAD
    multiple and no threshold. The screen decides how many to keep, which is the entire
    difference between an adaptive cut and a top-k with extra steps.

    ``item_embs`` is expected to be a **bounded candidate pool** — a top-B cosine
    horizon, not a corpus. The read is over the structure of the set it is given, so
    what is in the set is part of the question; handing it everything asks a different
    question and gets a different answer, and the SVD then sees an unbounded matrix.

    ``n <= 1`` returns every index. One candidate has no spectrum, so there is nothing
    to cut, and discarding the only evidence there is on the strength of a read that
    never happened would be worse than keeping it.
    """
    E = np.asarray(item_embs, dtype=np.float64)
    if len(E) <= 1:
        return np.arange(len(E))
    H = derive_heads(E) if heads is None else int(heads)
    power = signal_power(head_screen(E, query_emb, H))
    keep, _ = top_break(power)
    return np.where(keep)[0]


def select_with_info(
    candidate_embeddings,
    query_embedding,
    *,
    max_keep: int | None = None,
    min_horizon: int = 8,
) -> tuple[list[int], dict[str, Any]]:
    """:func:`select`, power-descending, with the evidence for what it did.

    Args:
        candidate_embeddings: ``(n, d)`` — a bounded cosine horizon.
        query_embedding: ``(d,)``.
        max_keep: a hard ceiling on the number kept, for a token budget. Reported as
            ``capped``: a truncated selection is not the cut's answer and must not be
            read as one.
        min_horizon: below this many candidates, decline and hand back everything with
            ``fell_back`` set. This is **caller policy, not a property of the read** —
            the engine's own gate is ``n >= 2``. 8 is the value the live pilot ships;
            it says how much structure a caller wants present before it trusts a cut,
            and a caller with a different appetite should pass a different number.

    Returns ``(kept_indices, info)``. ``info`` carries ``horizon``, ``heads_derived``,
    ``kept``, ``rel_gap``, ``separability``, ``found_gap``, ``fell_back``, ``capped``.

    ``separability`` is the evidence that there was anything to cut: eta^2 of the best
    two-population split of the signal power, 0.0 for one population. It discriminates
    better than ``rel_gap`` and not well enough to trust alone: the structured and
    structureless populations touch at the edges. Both are reported and neither is gated on,
    because a gate needs a threshold and a structureless pool scores too high to set one.

    ``found_gap`` is False when the lock kept everything — the cut ran and found no
    break — and that is a different fact from ``fell_back``, where it did not run.
    """
    E = np.asarray(candidate_embeddings, dtype=np.float64)
    n = len(E)
    info: dict[str, Any] = {
        "horizon": n, "heads_derived": 0, "kept": 0, "rel_gap": None,
        "separability": None, "found_gap": False, "fell_back": False, "capped": False,
    }
    if n == 0:
        return [], info
    if n < min_horizon:
        info["fell_back"] = True
        info["kept"] = n
        return list(range(n)), info

    heads = derive_heads(E)
    info["heads_derived"] = heads
    power = signal_power(head_screen(E, query_embedding, heads))
    keep_mask, rel_gap = top_break(power)
    info["rel_gap"] = float(rel_gap)
    info["separability"] = round(engine.separability(power), 4)
    order = np.argsort(power)[::-1]
    kept = [int(i) for i in order if keep_mask[i]]
    info["found_gap"] = len(kept) < n
    if max_keep is not None and len(kept) > max_keep:
        info["capped"] = True
        kept = kept[:max_keep]
    info["kept"] = len(kept)
    return kept, info

# ──────────────────────────────────────────────────────────────────────────
# The lock on plain cosine salience: the same lock, without the screen.
# ──────────────────────────────────────────────────────────────────────────


def cosine_salience(item_embs, query_emb) -> np.ndarray:
    """``(n,)`` — cosine of each candidate against the query, clipped at zero.

    Clipped rather than shifted. A negative cosine says the candidate points away from the
    query, and every such candidate is equally uninformative about where the spectrum
    stops; mapping them all to zero says that, where shifting to the minimum would spread
    them into a tail that the lock then has to look past.
    """
    E = np.asarray(item_embs, dtype=np.float64)
    q = np.asarray(query_emb, dtype=np.float64).ravel()
    if E.ndim != 2:
        raise CutUnavailable(f"candidates must be (n, d); got shape {E.shape}")
    if E.shape[1] != q.size:
        raise CutUnavailable(
            f"candidates are {E.shape[1]}-dimensional and the query is {q.size}"
        )
    en = np.linalg.norm(E, axis=1)
    qn = float(np.linalg.norm(q))
    if qn == 0.0 or not np.isfinite(qn) or not np.isfinite(en).all() or (en == 0).any():
        raise CutUnavailable("a zero-length or non-finite vector has no direction to compare")
    return np.clip((E @ q) / (en * qn), 0.0, None)


def select_by_cosine(item_embs, query_emb) -> np.ndarray:
    """Which of a candidate horizon to keep for this query. Returns ascending indices.

    The count is read from the candidates' own similarity spectrum: cut at the largest
    relative gap at or above the median (:func:`~entroptics_llm.lock.top_break`). No fixed
    *k*, no similarity threshold, nothing fitted to a corpus, and no additional model call —
    the similarities are ones the retriever already computed.

    ``n <= 1`` returns every index. One candidate has no spectrum, and discarding the only
    evidence there is on the strength of a read that never happened would be worse than
    keeping it.

    What it is for
    --------------
    The best fixed *k* is a different number on each corpus, so a service that answers
    against more than one of them, or against a corpus whose content moves, has no single
    right constant to ship. This reads the number per query instead, and the measured
    consequence is a **bound on what that costs**: across five BEIR corpora the best single
    constant gives up as much as 0.086 F1 against a corpus's own tuned *k*, where this gives
    up at most 0.019. ``research/PAPER.md`` has the measurement.

    Where a constant is better
    --------------------------
    One corpus, stable, with relevance labels to sweep against. Find the *k* and ship it —
    it is cheaper than this and at least as good. What this is for is the case where that
    sweep is not available.
    """
    E = np.asarray(item_embs, dtype=np.float64)
    if E.ndim != 2 or len(E) <= 1:
        return np.arange(len(E))
    keep, _ = top_break(cosine_salience(E, query_emb))
    return np.where(keep)[0]
