"""Where a salience spectrum stops.

Given a non-negative score per candidate, decide how many to keep. The answer is read from
the spectrum's own shape: sort descending, take the ratio between each value and the next,
and cut at the largest one. Nothing is fitted, nothing is swept, and the same spectrum in
different units cuts in the same place, because a ratio is scale-free.

Two locks, for two shapes of question.

``gap_split``  cut at the largest relative gap anywhere in the spectrum.
``top_break``  isolate the top cluster: the largest relative gap at or above the median.
               This is the one a retrieval budget wants, and the one every published
               number in ``research/`` is measured on.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gap_split", "top_break"]

#: A denominator floor for the consecutive ratios. Salience values reaching zero are
#: ordinary — a clipped cosine does it whenever a candidate points away from the query —
#: and a ratio against zero carries no information about where the spectrum stops.
_RATIO_FLOOR = 1e-12


def _salience(scores, where: str) -> np.ndarray:
    """Validate and flatten. Raises rather than returning a number it cannot stand behind."""
    from .engine import CutUnavailable

    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size and not np.isfinite(s).all():
        bad = int((~np.isfinite(s)).sum())
        raise CutUnavailable(
            f"{where}: {bad} of {s.size} values are NaN or infinite. A spectrum with a hole "
            f"in it has no break to find, and 'keep everything' would report that as a decision."
        )
    if s.size and s.min() < 0.0:
        raise CutUnavailable(
            f"{where}: the salience must be non-negative; the minimum is {s.min():.6g}. "
            f"A multiplicative gap between negative values is not a drop. Shift the "
            f"spectrum to its own minimum first if the sign carries meaning."
        )
    return s


def _tie_tol(s: np.ndarray) -> float:
    """The relative width within which two values of ``s`` are the same computed number.

    ``2 * n * eps``: a sum of at most ``n`` terms carries a relative forward error bounded
    by ``n * eps``, and a ratio of two such values carries twice that. The epsilon is read
    off the array's own dtype rather than asserted, so the tolerance follows the precision
    the caller is working in.

    It is computed on the whole spectrum. :func:`top_break` narrows to a region before
    locking, and the values inside that region were produced by the same arithmetic as the
    ones outside it, so the tolerance belongs to the spectrum rather than to the slice.
    """
    from .engine import machine_eps
    return 2.0 * max(1, s.size) * machine_eps(s)


def _gap_split(s: np.ndarray, tol: float):
    """:func:`gap_split` on a validated spectrum with a tolerance supplied by the caller."""
    n = s.size
    if n <= 1:
        return np.ones(n, dtype=bool), 1.0
    order = np.argsort(s)[::-1]
    sd = s[order]
    ratios = sd[:-1] / np.clip(sd[1:], _RATIO_FLOOR, None)
    cut = int(np.argmax(ratios))
    keep = np.zeros(n, dtype=bool)
    if ratios[cut] <= 1.0 + tol:                # no drop that clears the round-off
        keep[:] = True
        return keep, 1.0
    keep[order[: cut + 1]] = True
    return keep, float(ratios[cut])


def gap_split(scores):
    """Cut after the largest relative gap in a salience spectrum.

    Sort descending, take the consecutive top-over-next ratios, cut at the biggest
    multiplicative drop. There is no threshold, no MAD multiple, no significance level and
    no keep-fraction cap.

    Returns ``(keep_mask, rel_gap)`` over the original items, where ``rel_gap`` is the size
    of the break. It is a statistic rather than a verdict: whether a break is *real* is a
    question about a reference distribution, not about one spectrum.

    A spectrum with no drop keeps everything and reports ``rel_gap`` 1.0. "No drop" means
    no consecutive ratio clears 1 by more than the round-off of the arithmetic that
    produced the values — a ratio of ``1 + 4e-15`` between two identical rows is the same
    computed number twice, not a gap.
    """
    s = _salience(scores, "gap_split")
    return _gap_split(s, _tie_tol(s))


def top_break(scores):
    """Isolate the top cluster: the largest relative gap at or above the median.

    The median is a data-derived floor. It keeps the largest ratio from being won by a pair
    of adjacent near-zeros in the tail, where a ratio is large for no reason.

    The region is extended by one element below the floor. The break that ends a top
    cluster sits *between* the cluster and what follows it, which is the boundary the floor
    would otherwise exclude: on ``[9, 8, 8, 1, 1, 1, 1, 1]`` the strictly-above-median
    values offer only a near-unit ratio between neighbours and cut after the 9, keeping one of
    an obvious three, while the real break of 8 → 1 lies just across the floor. The
    boundary element makes it visible and is never itself kept.

    Returns ``(keep_mask, rel_gap)``. Two degenerate spectra have defined answers: a flat
    one, with nothing measurably above its median, keeps everything and reports ``rel_gap``
    1.0 — the read ran and found no break; and a tied top group is kept whole, because
    there is no drop inside it to cut at and one arbitrary member is not an answer.
    """
    s = _salience(scores, "top_break")
    n = s.size
    keep = np.zeros(n, dtype=bool)
    if n <= 1:
        keep[:] = True
        return keep, 1.0
    tol = _tie_tol(s)
    order = np.argsort(s)[::-1]
    med = float(np.median(s))
    # Measurably above the median. A value sitting a few last bits over it is the median,
    # and admitting it opens a one- or two-element region whose only break is round-off.
    n_above = int((s[order] > med * (1.0 + tol)).sum())
    if n_above == 0:
        keep[:] = True
        return keep, 1.0

    region = order[: min(n_above + 1, n)]              # + the boundary element
    keep_local, rel_gap = _gap_split(s[region], tol)   # the whole spectrum's tolerance
    n_keep = min(int(keep_local.sum()), n_above)       # the boundary element is never kept
    keep[order[:n_keep]] = True
    return keep, rel_gap
