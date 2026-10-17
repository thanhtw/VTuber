"""Weighted denominators, channel bootstrap CIs, and explicit permutation nulls."""
from itertools import combinations
import numpy as np

PAIRS = list(combinations(range(9), 2))


def finite(value):
    return float(value) if np.isfinite(value) else None


def bh_adjust(values):
    output = [None] * len(values)
    valid = [(i, p) for i, p in enumerate(values) if p is not None and np.isfinite(p)]
    valid.sort(key=lambda pair: pair[1])
    previous, total = 1., len(valid)
    for rank in range(total, 0, -1):
        i, p = valid[rank-1]
        previous = min(previous, p*total/rank)
        output[i] = previous
    return output


def bootstrap_weights(channels, count, rng):
    unique, inverse = np.unique(np.asarray(channels), return_inverse=True)
    if len(unique) < 2:
        return None
    multiplicities = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=count)
    return multiplicities[:, inverse]


def proportion(values, weights, channels, samples=1000, seed=42):
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    active = weights > 0
    values, weights = values[active], weights[active]
    channels = list(np.asarray(channels)[active])
    denominator = float(weights.sum())
    estimate = float(weights @ values / denominator) if denominator else None
    unique, inverse = np.unique(np.asarray(channels), return_inverse=True)
    lo, hi = None, None
    if len(unique) >= 2 and denominator:
        rng = np.random.default_rng(seed)
        multiplicities = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=samples)
        group_numerator = np.bincount(inverse, weights=weights*values)
        group_denominator = np.bincount(inverse, weights=weights)
        totals = multiplicities @ group_denominator
        estimates = np.divide(multiplicities @ group_numerator, totals, out=np.full(samples, np.nan), where=totals > 0)
        valid = estimates[np.isfinite(estimates)]
        if len(valid):
            lo, hi = map(float, np.quantile(valid, [.025, .975]))
    return {"weighted_numerator": float(weights @ values), "weighted_denominator": denominator,
            "estimate": estimate, "ci_low": lo, "ci_high": hi,
            "channels": len(set(channels)), "bootstrap_unit": "channel"}


def _totals(x, weights):
    total = weights.sum()
    marginal = weights @ x
    joint = np.array([weights @ (x[:, a]*x[:, b]) for a, b in PAIRS])
    expected = np.array([marginal[a]*marginal[b]/total if total else np.nan for a, b in PAIRS])
    ratio = np.divide(joint, expected, out=np.full(36, np.nan), where=expected > 0)
    return total, marginal, joint, expected, ratio


def fixed_margin_swap(x, groups, rng, attempts):
    """Checkerboard swaps preserve each comment's degree and within-channel aspect margins."""
    result = x.copy()
    eligible = [indices for indices in groups if len(indices) >= 2]
    accepted = 0
    if not eligible:
        return result, accepted
    for _ in range(attempts):
        indices = eligible[int(rng.integers(len(eligible)))]
        row1, row2 = rng.choice(indices, size=2, replace=False)
        left = np.flatnonzero((result[row1] == 1) & (result[row2] == 0))
        right = np.flatnonzero((result[row1] == 0) & (result[row2] == 1))
        if len(left) and len(right):
            a, b = rng.choice(left), rng.choice(right)
            result[row1, a], result[row1, b] = 0, 1
            result[row2, a], result[row2, b] = 1, 0
            accepted += 1
    return result, accepted


def associations(x, channels, weights=None, bootstrap=1000, permutations=1000, seed=42,
                 null="within_channel_column_permutation"):
    x = np.asarray(x, dtype=int).reshape((-1, 9))
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if len(x) != len(channels) or len(x) != len(weights) or np.any(weights < 0):
        raise ValueError("Incidence, channel, and nonnegative weight lengths must agree.")
    if not len(x) or not weights.sum():
        return []
    rng = np.random.default_rng(seed)
    total, marginal, observed, expected, ratios = _totals(x, weights)
    intervals = np.full((2, 36), np.nan)
    sampled = bootstrap_ratios(x, channels, weights, bootstrap, seed)
    if sampled is not None:
        for j in range(36):
            valid = sampled[:, j][np.isfinite(sampled[:, j])]
            if len(valid):
                intervals[:, j] = np.quantile(valid, [.025, .975])
    groups = [np.flatnonzero(np.asarray(channels) == key) for key in sorted(set(channels))]
    null_counts, movement = [], 0
    for _ in range(permutations):
        if null == "fixed_margin_within_channel":
            permuted, accepted = fixed_margin_swap(x, groups, rng, max(100, 5*len(x)))
            movement += accepted
        else:
            permuted = x.copy()
            for indices in groups:
                for j in range(9):
                    permuted[indices, j] = rng.permutation(x[indices, j])
            movement += int(np.any(permuted != x))
        null_counts.append(_totals(permuted, weights)[2])
    null_counts = np.asarray(null_counts)
    rows = []
    for j, (a, b) in enumerate(PAIRS):
        draws = null_counts[:, j]
        center = float(draws.mean())
        variable = bool(np.ptp(draws) > 0)
        p = (1 + np.count_nonzero(np.abs(draws-center) >= abs(observed[j]-center)-1e-12))/(permutations+1) if variable else None
        rows.append({"aspect_a": f"A{a+1}", "aspect_b": f"A{b+1}",
                     "comments": len(x), "channels": len(set(channels)),
                     "weighted_comments": float(total), "aspect_a_mentions": float(marginal[a]),
                     "aspect_b_mentions": float(marginal[b]), "observed": float(observed[j]),
                     "cooccurrence_proportion": float(observed[j]/total),
                     "expected_independence": finite(expected[j]), "observed_expected_ratio": finite(ratios[j]),
                     "ratio_ci_low": finite(intervals[0, j]), "ratio_ci_high": finite(intervals[1, j]),
                     "permutation_expected": center, "p_two_sided": p,
                     "null": null, "null_variable": variable, "null_movement": movement,
                     "bootstrap_samples": bootstrap, "permutations": permutations,
                     "interpretation": "provisional_observable_association"})
    for row, q in zip(rows, bh_adjust([r['p_two_sided'] for r in rows])):
        row['q_within_network'] = q
    return rows


def bootstrap_ratios(x, channels, weights, count, seed):
    unique = sorted(set(channels))
    if len(unique) < 2:
        return None
    stats = []
    for channel in unique:
        mask = np.asarray(channels) == channel
        total, marginal, joint, _, _ = _totals(x[mask], weights[mask])
        stats.append(np.concatenate(([total], marginal, joint)))
    rng = np.random.default_rng(seed)
    m = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=count)
    sums = m @ np.asarray(stats)
    total, marginal, joint = sums[:, 0], sums[:, 1:10], sums[:, 10:]
    expected = np.column_stack([np.divide(marginal[:, a]*marginal[:, b], total, out=np.full(count, np.nan), where=total > 0) for a, b in PAIRS])
    return np.divide(joint, expected, out=np.full_like(joint, np.nan), where=expected > 0)
