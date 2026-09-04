"""The lock: where a salience spectrum stops."""
import numpy as np
import pytest

from entroptics_llm.engine import CutUnavailable
from entroptics_llm.lock import gap_split, top_break


def test_gap_split_cuts_at_the_largest_relative_drop():
    keep, gap = gap_split([10.0, 9.0, 8.0, 0.5, 0.4])
    assert list(np.where(keep)[0]) == [0, 1, 2]
    assert gap == pytest.approx(16.0)


def test_gap_split_is_scale_free():
    s = np.array([10.0, 9.0, 8.0, 0.5, 0.4])
    a, ga = gap_split(s)
    b, gb = gap_split(s * 1e6)
    assert np.array_equal(a, b) and ga == pytest.approx(gb)


def test_gap_split_is_order_independent():
    s = np.array([0.4, 8.0, 10.0, 0.5, 9.0])
    keep, _ = gap_split(s)
    assert set(np.where(keep)[0]) == {1, 2, 4}


def test_top_break_ignores_a_tail_ratio_gap_splits_would_take():
    # The largest relative gap in the raw spectrum is in the tail (1e-3 -> 1e-9), which
    # is exactly the artefact the above-median restriction exists to refuse. top_break
    # keeps the {10, 9} cluster: the break it locks on is 9 -> 1, which is visible only
    # because the region carries one element below the median floor.
    s = np.array([10.0, 9.0, 1.0, 1e-3, 1e-9])
    assert list(np.where(gap_split(s)[0])[0]) == [0, 1, 2, 3]
    assert list(np.where(top_break(s)[0])[0]) == [0, 1]


def test_a_flat_spectrum_keeps_everything_rather_than_one_arbitrary_item():
    """No break exists, so none is reported. See the top_break docstring."""
    keep, gap = top_break(np.ones(9))
    assert keep.all() and gap == pytest.approx(1.0)


def test_a_lone_peak_is_kept_and_reports_how_far_clear_it_stood():
    keep, gap = top_break(np.array([9.0, 1.0, 1.0, 1.0, 1.0]))
    assert list(np.where(keep)[0]) == [0] and gap == pytest.approx(9.0)


def test_degenerate_sizes():
    for n in (0, 1):
        keep, gap = top_break(np.ones(n))
        assert keep.sum() == n and gap == 1.0
    keep, _ = top_break(np.array([5.0, 1.0]))
    assert list(np.where(keep)[0]) == [0]


def test_zero_tail_reports_a_finite_gap_and_cuts_above_it():
    keep, gap = gap_split([3.0, 2.0, 0.0, 0.0])
    assert list(np.where(keep)[0]) == [0, 1]
    assert np.isfinite(gap)


# ── the boundary element ──────────────────────────────────────────────────────

def test_the_break_that_ends_a_cluster_lies_on_the_median_boundary():
    """Why the region carries one element below the floor.

    Restricted to strictly-above-median values, [9,8,8,1,1,1,1,1] offers only the ratios
    1.125 and 1.000 and cuts after the 9 — one item of an obvious three. The real break,
    8 -> 1 at a ratio of 8, sits on the other side of the floor."""
    for spectrum, want in (([9, 8, 8, 1, 1, 1, 1, 1], 3),
                           ([9, 9, 9, 5, 5, 5, 1, 1], 3),
                           ([5, 5, 5, 5, 1, 1, 1, 1], 4),
                           ([7, 7, 1, 1], 2),
                           ([9, 1, 1, 1, 1], 1)):
        keep, _ = top_break(np.array(spectrum, float))
        assert int(keep.sum()) == want, (spectrum, int(keep.sum()), want)


def test_the_boundary_element_is_never_itself_kept():
    keep, _ = top_break(np.array([9, 8, 8, 1, 1, 1, 1, 1], float))
    assert not keep[3]                       # the 1 that made the break visible


# ── no drop is not a small drop ───────────────────────────────────────────────

def test_a_tied_group_is_kept_whole_not_cut_to_one_arbitrary_member():
    for spectrum in ([3, 3, 3], [1, 1], [0, 0, 0, 0], [2.5] * 7):
        keep, gap = gap_split(np.array(spectrum, float))
        assert keep.all(), spectrum
        assert gap == pytest.approx(1.0)


def test_a_ratio_within_the_arithmetics_own_round_off_is_not_a_break():
    """Identical candidate rows give powers differing in the last bits. A ratio of
    1 + 4e-15 was cutting twenty identical items down to one."""
    base = 1.0015940782057581e-3
    s = np.full(20, base)
    s[:4] = np.nextafter(base, np.inf)       # a few ulps up: the same computed number
    keep, gap = top_break(s)
    assert keep.all() and gap == pytest.approx(1.0)


def test_the_tolerance_is_read_off_the_dtype_not_asserted():
    from entroptics_llm.engine import machine_eps
    assert machine_eps(np.zeros(3, dtype=np.float64)) == np.finfo(np.float64).eps
    assert machine_eps(np.zeros(3, dtype=np.float32)) == np.finfo(np.float32).eps


def test_a_real_break_still_survives_the_tolerance():
    keep, gap = gap_split(np.array([10.0, 9.0, 1.0], float))
    assert list(np.where(keep)[0]) == [0, 1] and gap == pytest.approx(9.0)


# ── input the lock cannot stand behind ────────────────────────────────────────

def test_negative_salience_is_refused_rather_than_scored():
    """The previous version returned -1e12 as a "gap" on a descending negative series."""
    for bad in ([-1, -2, -3, -4], [5, 1, -1, -5], [-0.1, -0.2, -10]):
        for fn in (gap_split, top_break):
            with pytest.raises(CutUnavailable, match="non-negative"):
                fn(np.array(bad, float))


def test_non_finite_salience_is_refused_rather_than_kept_whole():
    """[1, nan, 3] used to return "keep everything, gap 1.000" — a decision, from a hole."""
    for bad in ([1.0, np.nan, 3.0], [1.0, np.inf, 3.0], [np.nan] * 4):
        for fn in (gap_split, top_break):
            with pytest.raises(CutUnavailable, match="NaN or infinite"):
                fn(np.array(bad, float))


def test_a_shifted_signed_spectrum_is_the_documented_way_through():
    """Shifting a signed spectrum to non-negative is what the error message tells callers
    to do, and this pins that the documented route actually works."""
    signed = np.array([-1.0, -2.0, -9.0, -9.5], float)
    keep, _ = top_break(signed - signed.min())
    assert keep.any() and not keep.all()
