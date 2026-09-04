"""The one seam onto ``entroptics``.

Every spectral read in this package arrives here. No other module imports
``entroptics``, and ``tests/test_no_shadow.py`` enforces that, because the way this
family of code goes wrong is not a wrong formula — it is two copies of the right one
drifting a percent apart and neither being wrong enough to notice.

What the engine is asked for
----------------------------
=========================  ============================================================
``resolved_rank(M)``       how many directions ``M`` really has: singular values above
                           the derived noise floor. Whitened, native resolution.
``raw_signal_power(W)``    each row's energy in the top-``K`` left singular modes of
                           the **raw** screen, ``K`` from ``resolved_rank``. The
                           retrieval path — see "two substrates" below.
``occupancy(M)``           ``2^{H_sv}/n``, the fraction of modes carrying the energy.
``focus(E)``               ``sigma_1^2 / M``, the power fraction on the leading axis.
=========================  ============================================================

Two substrates, read two ways
------------------------------------------
The library's resolved read whitens each feature channel to a common noise scale
before decomposing, which is right when the channels are *incommensurate* — a KV
head's ``head_dim`` coordinates carry arbitrary learned scales, and a channel that
happens to be large is not thereby more informative.

A retrieval head screen is the opposite case. ``W[item, head]`` is a per-head cosine:
every entry is already in the same units, on the same bounded interval, and the
per-head magnitude *is* the relevance. Whitening it would divide exactly the signal
out. So ``raw_signal_power`` takes its modes off the raw screen.

It still takes its *count* from the whitened frame, because the noise floor is a
statement about unit-noise channels and is not meaningful otherwise. That mix — rank
from the whitened frame, modes from the raw one — is what the shipped implementation
does and what the measured retrieval win was measured on. It is stated here rather
than left to be discovered in the arithmetic.

Where this sits relative to the other copies
--------------------------------------------
``mantle.search.beacon.engine`` is a numpy-only reimplementation of the same reads,
carried in the Agience tree so a store can ship without this dependency. The two are
not assumed to agree; ``tests/test_conformance_mantle.py`` measures it, and the one
place they genuinely diverge is recorded there with the measured size: a channel of
measured zeros is a mode the record HAS, which ``occupancy`` counts in its denominator
and mantle's ``occupancy_fraction`` drops as a dead line.
"""
from __future__ import annotations

import numpy as np

from entroptics import resolved_batch
from entroptics.reads import concentration as _concentration
from entroptics.reads import phi as _phi

__all__ = [
    "CutUnavailable",
    "resolved_rank",
    "raw_signal_power",
    "occupancy",
    "focus",
    "noise_floor",
    "machine_eps",
    "separability",
]

#: Passed to every read as the noise floor's false-alarm level. It is the library's own
#: default and the only externally-set number in this package: not a tuned constant but
#: a stated tolerance — how often the floor is allowed to admit a mode that is noise.
FAR = 0.05


class CutUnavailable(RuntimeError):
    """A read could not be taken. Callers keep a fixed-cut fallback path.

    NOT a "no signal" result. A read that happened and found one coherent direction
    returns 1; a read that did not happen raises. Conflating the two is how a fallback
    becomes the product — every caller that catches this must do something
    visibly different from what it does on a rank of 1.
    """


def _frame(M) -> np.ndarray:
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2:
        raise CutUnavailable(f"a screen must be 2-D; got shape {A.shape}")
    if A.size == 0 or min(A.shape) < 2:
        raise CutUnavailable(f"a {A.shape} frame has no spectrum to read")
    return A


def _read(M, *, energy: bool = False, basis: bool = False):
    """One resolved read of a single frame, at native feature resolution.

    ``fold=False`` throughout. The feature axis here is always an unordered learned
    basis — embedding coordinates, a head's ``head_dim``, or the head index of a
    query-relative screen — and the library's area-average fold is valid only for a
    dense smooth continuum. ``fold="auto"`` would reach the same conclusion per frame;
    pinning it says so once instead of relying on it every call.
    """
    A = _frame(M)
    try:
        return resolved_batch(A[None], fold=False, far=FAR, energy=energy, basis=basis)
    except Exception as exc:                              # a genuinely unreadable frame
        raise CutUnavailable(f"resolved read failed on a {A.shape} frame: {exc!r}") from exc


def resolved_rank(M) -> int:
    """How many directions ``M`` has: singular values above the derived noise floor.

    Floored at 1. A caller here builds a subspace out of the answer immediately and
    cannot defer, so a genuine count of 0 becomes the coarsest usable basis rather
    than an empty one. The floor is this package's policy, not the engine's — the
    engine reports 0 when it means 0, which is why this wrapper is where the floor
    lives and not somewhere it could be applied twice.
    """
    return max(1, int(_read(M).K_signal[0]))


def raw_signal_power(W) -> np.ndarray:
    """``(n,)`` — each row's energy in the top-``K`` left singular modes of the raw ``W``.

    ``K = resolved_rank(W)``, read on the whitened frame; the modes are taken off ``W``
    as given. See the module docstring for why this substrate is not whitened.

    A row aligned with the coherent structure the screen actually has carries the
    power; a row that matches weakly and evenly does not. That is the whole difference
    between this and a row-sum, and therefore between an adaptive cut and a re-ranking.
    """
    A = _frame(W)
    k = min(resolved_rank(A), min(A.shape))
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    return ((U[:, :k] * S[:k]) ** 2).sum(1)


def occupancy(M) -> float:
    """``2^{H_sv}/n`` in (0, 1] — the fraction of ``M``'s modes that carry the energy.

    Read on the block as recorded. A constant level is a mode and is counted as one.

    This read and ``mantle.search.beacon.engine.occupancy_fraction`` agree exactly —
    measured over 60 real retrieval horizons, a maximum absolute difference of 0.0.
    They diverged until 2026-09-02, when the library still centred before decomposing:
    on a set of unit direction vectors the shared mean direction is the topic the set
    is about, so de-meaning deleted the dominant mode and roughly doubled the fill.
    That reached this package as twice the head count, and
    therefore as a different cut.

    NaN for a frame carrying no power: there is no fraction of active modes to report
    when no mode is active, and reporting ``1/n`` instead would be indistinguishable
    from a perfect single mode.
    """
    return float(_phi(_frame(M)))


def focus(E) -> float:
    """``sigma_1^2 / M`` in (0, 1] — the power fraction on the set's leading axis.

    The bounded MAGNITUDE companion to the rank read: :func:`resolved_rank` answers
    whether structure is present, this answers how concentrated it is. 1 = the set
    points one way; toward 0 = isotropic. Axial, so an antipodal cloud still reads ~1.

    A diagnostic rather than a decision. The concentration of a kept set tracks cut
    reliability only weakly, and the head screen's own concentration runs weakly the
    other way: a query matching everything is less discriminable. Use it to rank or
    compare sets, or to report a confidence beside a selection, and read the cut
    itself from the lock.
    """
    A = np.asarray(E, dtype=np.float64)
    if A.ndim != 2 or len(A) < 1:
        return 0.0
    return float(_concentration(A).focus)


def separability(salience) -> float:
    """``eta^2`` in [0, 1] — how cleanly a salience profile falls into two populations.

    0.0 means one population and nothing to cut at; toward 1.0 means two groups plainly
    apart. Delegates to ``entroptics.level_edge``.

    This is the **whether**, and the lock's ``rel_gap`` is the **where**. They are not
    interchangeable and neither substitutes for the other. Measured on 50 structured
    against 50 structureless candidate pools, as a detector of "is there anything here":
    this separates structured from structureless pools better than ``rel_gap``, whose
    ranges cross more broadly. Better, and not a solution — a structureless pool still
    scores well above zero, so this
    is evidence to report and not a gate to trust. It also measures *cleanness*, not
    distance: an exact two-level profile reads 1.000 whatever the ratio across it, so a
    caller who wants the size of the break still wants ``rel_gap``.

    Report it beside a selection. It is not a gate here, because turning it into one means
    choosing a threshold and that is the thing this package does not do.
    """
    from entroptics import level_edge
    return float(level_edge(np.asarray(salience, dtype=np.float64)).separability)


def machine_eps(ref) -> float:
    """The working dtype's machine epsilon — the smallest relative difference the arithmetic
    can represent, read off the array rather than asserted.

    Delegates to ``entroptics.entropy.macheps``, which is the library's own answer and
    follows the backend and the compute precision. It is exposed through this seam because
    :mod:`entroptics_llm.lock` needs it — a ratio of two computed numbers is not a drop
    unless it clears the round-off of the computation that produced them — and the lock
    does not import the library.

    This is the same construction as the library's ``resolution_floor`` (``typical * eps``,
    derived on both sides: the scale from the data, the epsilon from the dtype). What the
    lock adds is the term count, since its inputs are sums.
    """
    from entroptics.entropy import macheps
    return float(macheps(np, np.asarray(ref)))


def noise_floor(M) -> float:
    """The singular-value floor ``resolved_rank`` counted against, on the whitened frame.

    Exposed for reporting — "the top mode stood this far above the floor" is the
    evidence behind a rank, and a rank without it is a number with no error bar.
    """
    return float(_read(M).noise_floor[0])
