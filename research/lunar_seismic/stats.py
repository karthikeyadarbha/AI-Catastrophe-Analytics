"""Statistical tests for tidal / astrological modulation of earthquakes.

* Schuster's test  -- the standard test in the tidal-triggering literature for
  whether a set of phases on a circle departs from uniformity. p = exp(-R^2/N).
* Monte-Carlo test -- compares an observed statistic (e.g. the mean of a feature
  over all mainshocks) to its distribution across random-time null replicates.
* Benjamini-Hochberg -- controls the false-discovery rate across the many tests,
  so an exploratory sweep does not report chance hits as findings.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Schuster test (circular uniformity)
# --------------------------------------------------------------------------- #
def schuster_test(phases_deg: np.ndarray) -> Dict[str, float]:
    """Schuster's test for uniformity of angles on a circle.

    Args:
        phases_deg: phases in degrees (any range; taken mod 360).

    Returns:
        p       -- probability the observed concentration arose from a uniform
                   distribution (small = significant clustering).
        rbar    -- mean resultant length in [0, 1] (effect size / concentration).
        phase   -- preferred (mean) phase in degrees.
        n       -- sample size.
    """
    ph = np.radians(np.asarray(phases_deg, float) % 360.0)
    n = ph.size
    c = np.cos(ph).sum()
    s = np.sin(ph).sum()
    R2 = c * c + s * s
    p = float(np.exp(-R2 / n))
    return {
        "p": p,
        "rbar": float(np.sqrt(R2) / n),
        "phase": float(np.degrees(np.arctan2(s, c)) % 360.0),
        "n": int(n),
    }


# --------------------------------------------------------------------------- #
# Monte-Carlo test against random-time null replicates
# --------------------------------------------------------------------------- #
@dataclass
class McResult:
    obs_mean: float
    null_mean: float
    null_std: float
    z: float           # standardized effect size (obs - null_mean) / null_std
    p_two: float       # two-sided empirical p
    p_greater: float   # one-sided: observed mean is *higher* than null
    p_less: float      # one-sided: observed mean is *lower* than null
    n: int


def mc_test(observed: np.ndarray, null_matrix: np.ndarray) -> McResult:
    """Compare an observed feature to random-time null replicates.

    Args:
        observed: feature values for the real mainshocks, shape (n_events,).
        null_matrix: null feature values, shape (n_events, k). Each column is a
            full-catalog replicate at random times.
    """
    observed = np.asarray(observed, float)
    obs_mean = float(np.mean(observed))
    replicate_means = null_matrix.mean(axis=0)  # shape (k,)
    null_mean = float(replicate_means.mean())
    null_std = float(replicate_means.std(ddof=1))
    k = replicate_means.size
    # +1 smoothing so p is never exactly 0.
    p_greater = float((np.sum(replicate_means >= obs_mean) + 1) / (k + 1))
    p_less = float((np.sum(replicate_means <= obs_mean) + 1) / (k + 1))
    p_two = float(min(1.0, 2.0 * min(p_greater, p_less)))
    z = (obs_mean - null_mean) / null_std if null_std > 0 else 0.0
    return McResult(obs_mean, null_mean, null_std, float(z),
                    p_two, p_greater, p_less, int(observed.size))


# --------------------------------------------------------------------------- #
# Categorical / count test against null replicates (chi-square style)
# --------------------------------------------------------------------------- #
def category_excess(observed_counts: np.ndarray, null_counts: np.ndarray):
    """Per-category observed vs expected counts and a pooled chi-square p.

    Args:
        observed_counts: shape (n_categories,).
        null_counts: shape (n_categories, k) counts per null replicate.

    Returns:
        (expected, ratio, chi2, p_mc) where ratio = observed / expected and
        p_mc is the fraction of replicates whose chi-square vs the null mean is
        at least the observed chi-square.
    """
    expected = null_counts.mean(axis=1)
    safe = np.where(expected > 0, expected, np.nan)
    ratio = observed_counts / safe
    chi2_obs = np.nansum((observed_counts - expected) ** 2 / safe)
    chi2_null = np.nansum((null_counts - expected[:, None]) ** 2 / safe[:, None], axis=0)
    k = null_counts.shape[1]
    p_mc = float((np.sum(chi2_null >= chi2_obs) + 1) / (k + 1))
    return expected, ratio, float(chi2_obs), p_mc


# --------------------------------------------------------------------------- #
# Multiple-testing correction
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(pvalues, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downward
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, float)
    q[order] = np.clip(q_sorted, 0, 1)
    return q
