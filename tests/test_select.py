"""`select_by_cosine`: the lock applied to a retrieval horizon."""
import numpy as np
import pytest

from entroptics_llm import cosine_salience, select_by_cosine
from entroptics_llm.engine import CutUnavailable
from entroptics_llm.lock import top_break


def horizon(rng, n=40, d=64, planted=(2, 11, 29), strength=6.0):
    """A candidate set with a few items deliberately aligned to the query."""
    E = rng.standard_normal((n, d))
    q = rng.standard_normal(d)
    for i in planted:
        E[i] += strength * q
    return E, q, set(planted)


def test_it_is_the_lock_on_clipped_cosine_and_nothing_else():
    """The published numbers are measured on exactly this composition. If it drifts, the
    figures in research/ stop describing what the package does."""
    rng = np.random.default_rng(5)
    E, q, _ = horizon(rng, planted=(4,))
    sims = (E @ q) / (np.linalg.norm(E, axis=1) * np.linalg.norm(q))
    keep, _ = top_break(np.clip(sims, 0.0, None))
    assert np.array_equal(select_by_cosine(E, q), np.where(keep)[0])


def test_it_keeps_the_planted_items_and_few_others():
    rng = np.random.default_rng(6)
    E, q, planted = horizon(rng)
    kept = set(select_by_cosine(E, q).tolist())
    assert planted <= kept, kept
    assert len(kept) < 10, len(kept)


@pytest.mark.parametrize("planted", [8, 12])
def test_it_recovers_the_planted_count_exactly(planted):
    """The property the whole construction is for: told nothing about how many items were
    planted in a 40-candidate horizon, it returns exactly that many. Twelve seeds, no
    misses, for a group of eight or more."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        E, q, _ = horizon(rng, planted=tuple(range(planted)))
        kept = select_by_cosine(E, q)
        assert len(kept) == planted, (seed, len(kept))
        assert set(kept.tolist()) == set(range(planted)), seed


@pytest.mark.parametrize("planted,exact_at_least", [(2, 8), (3, 8), (5, 9)])
def test_smaller_groups_are_recovered_most_of_the_time_and_never_wildly(planted,
                                                                       exact_at_least):
    """A small group is a fainter break, and on some draws there is no drop that clears the
    round-off of the arithmetic. The read then keeps the above-median half rather than
    inventing a cut, so a miss is bounded by half the horizon instead of arbitrary."""
    counts = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        E, q, _ = horizon(rng, planted=tuple(range(planted)))
        kept = select_by_cosine(E, q)
        counts.append(len(kept))
        assert len(kept) == planted or len(kept) <= 20, (seed, len(kept))
    assert sum(1 for c in counts if c == planted) >= exact_at_least, counts


def test_it_is_scale_free():
    """A ratio does not care about units, so scaling the embeddings changes nothing."""
    rng = np.random.default_rng(10)
    E, q, _ = horizon(rng)
    assert np.array_equal(select_by_cosine(E, q), select_by_cosine(E * 1000.0, q))
    assert np.array_equal(select_by_cosine(E, q), select_by_cosine(E, q * 0.001))


def test_a_negative_cosine_is_clipped_rather_than_shifted():
    """Every candidate pointing away from the query is equally uninformative about where
    the spectrum stops, so they collapse to one value instead of spreading into a tail."""
    rng = np.random.default_rng(11)
    E, q, _ = horizon(rng)
    s = cosine_salience(E, q)
    assert s.min() >= 0.0
    assert (s == 0.0).sum() >= 1


def test_one_candidate_is_returned_rather_than_cut():
    assert list(select_by_cosine(np.ones((1, 8)), np.ones(8))) == [0]
    assert list(select_by_cosine(np.zeros((0, 8)), np.ones(8))) == []


def test_a_read_that_cannot_happen_is_refused_not_guessed():
    rng = np.random.default_rng(12)
    E = rng.standard_normal((10, 16))
    E[3] = 0.0
    with pytest.raises(CutUnavailable, match="no direction"):
        select_by_cosine(E, rng.standard_normal(16))
    with pytest.raises(CutUnavailable, match="dimensional"):
        select_by_cosine(rng.standard_normal((10, 16)), rng.standard_normal(8))
    with pytest.raises(CutUnavailable, match=r"\(n, d\)"):
        cosine_salience(rng.standard_normal(16), rng.standard_normal(16))


def test_indices_come_back_in_ascending_order():
    """So a caller can slice a candidate list with them directly."""
    rng = np.random.default_rng(13)
    E, q, _ = horizon(rng)
    idx = select_by_cosine(E, q)
    assert list(idx) == sorted(idx)
