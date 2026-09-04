"""The two implementations of these reads agree exactly, and this pins that.

``mantle.search.beacon.engine`` is a numpy-only implementation of the same screen reads,
carried in the Agience tree so a store can ship without the ``entroptics`` dependency. Two
copies of one formula is two copies to keep in agreement, and a divergence between them
surfaces as retrieval that disagrees with itself rather than as an error.

Agreement is measured on retrieval-shaped frames -- unit rows sharing a mean direction,
because that direction is the topic -- rather than on random ones. The two populations
answer differently: a difference that is negligible on random frames reaches the caller as
a changed head count on real horizons, and so as a different cut.

Run it with mantle on the path::

    PYTHONPATH=<agience-mantle>/src pytest tests/test_conformance_mantle.py
"""
from __future__ import annotations

import numpy as np
import pytest

mantle = pytest.importorskip(
    "mantle.search.beacon.engine",
    reason="mantle is a sibling checkout; this file measures agreement with it",
)

from entroptics_llm import engine  # noqa: E402


def horizons(rng, n=40, count=25, d=64):
    """Retrieval-shaped frames: unit rows sharing a mean direction, which is the case the
    old estimate was never measured on and the case that matters."""
    out = []
    for _ in range(count):
        topic = rng.standard_normal(d)
        E = rng.standard_normal((n, d)) * 0.6 + topic
        out.append(E / np.linalg.norm(E, axis=1, keepdims=True))
    return out


def test_occupancy_agrees_exactly_on_retrieval_horizons():
    """Exact, not close. Both now read the block as recorded, so any difference is a bug."""
    worst = 0.0
    for E in horizons(np.random.default_rng(0)):
        worst = max(worst, abs(engine.occupancy(E) - float(mantle.occupancy_fraction(E))))
    assert worst == 0.0, f"max |difference| {worst:.3e}"


def test_occupancy_agrees_on_awkward_frames():
    """A heavy tail and a near-degenerate set: where two implementations of one formula drift
    first. Both are exact. The measured-zero channel is a real divergence and has its own
    test below."""
    rng = np.random.default_rng(1)
    rng.standard_normal((30, 24))                       # keep the draw order of the old test
    frames = [rng.standard_normal((30, 24)) * (rng.pareto(2.0, (30, 1)) + 1.0)]
    E = rng.standard_normal((30, 24))                   # near-degenerate
    E[10:] = E[0]
    frames.append(E)
    for E in frames:
        assert engine.occupancy(E) == pytest.approx(
            float(mantle.occupancy_fraction(E)), rel=0, abs=1e-12)


def test_a_measured_zero_channel_is_where_the_two_still_differ():
    """The one remaining divergence, located and attributed rather than averaged away.

    A channel of measured zeros contributes no singular value but is still a mode the record
    HAS. ``entroptics`` counts it in the denominator; ``mantle.occupancy_fraction`` drops it
    as a dead line. On a 30x24 frame with one zeroed channel that is 24 modes against 23, so
    mantle reads 0.6442 where the library reads 0.6174 -- the ratio 23/24 exactly.

    **The library is right and mantle is not**, by the companion paper's own rule: a channel
    that was measured and read zero is an observation of no power and counts, while a channel
    that was never measured is absent, and absence is not an observation of zero. Mantle's
    docstring claims dropping dead lines is what makes it agree with ``phi``; it is what makes
    it disagree. Pinned here so the direction and the size are on the record, and so a fix on
    the mantle side turns this test red rather than passing silently."""
    rng = np.random.default_rng(1)
    E = rng.standard_normal((30, 24))
    E[:, 5] = 0.0
    ours, theirs = engine.occupancy(E), float(mantle.occupancy_fraction(E))
    assert ours < theirs, (ours, theirs)
    assert ours / theirs == pytest.approx(23.0 / 24.0, rel=1e-6)


def test_a_constant_level_moves_both_the_same_way():
    """The property whose absence caused the drift: a constant level is a mode, both count
    it, and both therefore report a more coherent block when one is added."""
    rng = np.random.default_rng(2)
    E = rng.standard_normal((40, 32))
    base_e, base_m = engine.occupancy(E), float(mantle.occupancy_fraction(E))
    for offset in (5.0, 50.0):
        e, mm = engine.occupancy(E + offset), float(mantle.occupancy_fraction(E + offset))
        assert e < base_e and mm < base_m, (offset, e, base_e)
        assert e == pytest.approx(mm, rel=0, abs=1e-12)


def test_the_head_count_that_follows_from_it_agrees():
    """Agreement matters because it reaches the selection. The head count is
    ``round(occupancy * n)``, so a divergence in the fraction becomes a different screen."""
    from entroptics_llm.cut import derive_heads
    rng = np.random.default_rng(3)
    for E in horizons(rng, count=15):
        theirs = int(max(2, min(64, round(float(mantle.occupancy_fraction(
            E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9))) * E.shape[0]))))
        assert derive_heads(E) == theirs, (derive_heads(E), theirs)
