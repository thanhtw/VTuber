import numpy as np
import pytest
from youtube_nlp.research.statistics import associations, bh_adjust, fixed_margin_swap, proportion


def test_pair_counts_expected_denominator_and_channels():
    x = np.zeros((8, 9), dtype=int)
    x[:4, 0] = 1
    x[:4, 1] = 1
    rows = associations(x, ['a']*4 + ['b']*4, bootstrap=20, permutations=20, seed=4)
    pair = rows[0]
    assert len(rows) == 36
    assert pair['observed'] == 4
    assert pair['expected_independence'] == 2
    assert pair['observed_expected_ratio'] == 2
    assert pair['comments'] == 8  # Zero-aspect comments remain in the denominator.
    assert pair['channels'] == 2
    assert rows[-1]['observed_expected_ratio'] is None
    assert rows[-1]['p_two_sided'] is None


def test_one_channel_does_not_create_false_bootstrap_precision():
    x = np.ones((3, 9), dtype=int)
    row = associations(x, ['a']*3, bootstrap=10, permutations=10)[0]
    assert row['ratio_ci_low'] is None
    assert row['ratio_ci_high'] is None
    assert row['p_two_sided'] is None  # Degenerate null is not significant.


def test_weighted_prevalence_and_conditional_denominators():
    result = proportion([1, 0, 0], [.75, .25, 0], ['a', 'b', 'unused'], samples=20)
    assert result['estimate'] == .75
    assert result['weighted_denominator'] == 1
    assert result['channels'] == 2
    empty = proportion([], [], [], samples=10)
    assert empty['estimate'] is None


def test_checkerboard_preserves_row_and_channel_margins():
    x = np.array([[1,0,1,0], [0,1,0,1], [1,1,0,0], [0,0,1,1]])
    y, moves = fixed_margin_swap(x, [np.array([0,1]), np.array([2,3])], np.random.default_rng(2), 100)
    assert moves > 0
    np.testing.assert_array_equal(x.sum(axis=1), y.sum(axis=1))
    for ids in [[0,1],[2,3]]:
        np.testing.assert_array_equal(x[ids].sum(axis=0), y[ids].sum(axis=0))


def test_bh_adjustment_handles_missing_tests():
    assert bh_adjust([.01, .04, None, .03]) == pytest.approx([.03, .04, None, .04], nan_ok=True)
