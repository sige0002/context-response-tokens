"""Paired cluster bootstrap for fixed-model benchmark scores.

The estimand is the mean over original annotated examples. Clusters are sampled
with replacement and carry all their examples, so duplicate RGB videos with
different annotations retain their official weight without being independent.
This does not estimate training-seed variation or establish causation.
"""
import numpy as np


def paired_mean_interval(a, b, groups, repeats=10000, seed=17):
    a, b, groups = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(groups)
    if (a.ndim != 1 or b.shape != a.shape or groups.shape != a.shape or not len(a)
            or not np.isfinite(a).all() or not np.isfinite(b).all() or repeats < 100):
        raise ValueError("Expected finite, paired scores, matching groups, and at least 100 bootstrap draws")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("An uncertainty interval requires at least two independent clusters")
    count = np.bincount(inverse)
    difference = a - b
    sums = np.bincount(inverse, weights=difference)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(len(unique), size=len(unique))
        draws[i] = sums[selected].sum() / count[selected].sum()
    return dict(a_mean=float(a.mean()), b_mean=float(b.mean()),
        difference=float(difference.mean()), conditional_95_interval=np.quantile(draws, [.025, .975]).tolist(),
        examples=len(a), clusters=len(unique), bootstrap_repeats=repeats, bootstrap_seed=seed,
        estimation="Paired cluster percentile bootstrap of the example-weighted mean; conditional on fitted models")
