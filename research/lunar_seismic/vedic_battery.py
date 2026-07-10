"""Generic Vedic-astrology battery vs a matched random-time null.

This is the sidereal successor to :mod:`astro_pipeline`. Instead of a hand-listed
set of features it consumes whatever :class:`astro_engine.vedic.VedicFeatures`
emits -- every categorical feature is tested with a chi-square goodness-of-fit
and every boolean with a binomial test, all against a **matched random-time
null** (real epicenters, random times) and corrected together with
Benjamini-Hochberg FDR.

A "hit" here means only "worth a pre-registered, out-of-sample retest" -- never a
proven effect and certainly not causation. Two built-in guards keep us honest:

* the **Sun's sign** (= time of year) is a negative control -- a correct null
  model must leave it non-significant;
* the null is spatially matched, so seismic geography cannot masquerade as an
  astrological signal.

Because the feature count is large (hundreds), FDR keeps false positives bounded
but statistical power per feature is modest; restrict ``--include`` to sharpen a
specific hypothesis (e.g. ``--include declination`` for the tidal-declination
lead).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from astro_engine.vedic import VedicFeatures, FeatureSet

from .catalog import fetch_catalog
from .decluster import decluster
from .nulls import random_times
from .stats import benjamini_hochberg

_OUT = Path(__file__).resolve().parent / "outputs"
_SHALLOW_KM = 70.0


# -- null accumulation --------------------------------------------------------

class _NullAccumulator:
    """Streams pooled null counts so memory stays O(features), not O(samples)."""

    def __init__(self):
        self.cat_counts: dict = {}
        self.cat_valid: dict = {}
        self.flag_true: dict = {}
        self.flag_total: dict = {}

    def add(self, fs: FeatureSet) -> None:
        for name in fs.cat:
            counts, valid = fs.categorical_counts(name)
            if name not in self.cat_counts:
                self.cat_counts[name] = np.zeros_like(counts)
                self.cat_valid[name] = 0
            self.cat_counts[name] += counts
            self.cat_valid[name] += valid
        for name in fs.flag:
            t, n = fs.flag_count(name)
            self.flag_true[name] = self.flag_true.get(name, 0) + t
            self.flag_total[name] = self.flag_total.get(name, 0) + n

    def cat_prob(self, name: str) -> np.ndarray:
        v = self.cat_valid[name]
        return self.cat_counts[name] / v if v else self.cat_counts[name]

    def flag_prob(self, name: str) -> float:
        n = self.flag_total[name]
        return self.flag_true[name] / n if n else 0.0


def pooled_null(vf: VedicFeatures, catalog, k, seed, chunk=40_000) -> _NullAccumulator:
    rng = np.random.default_rng(seed)
    n = len(catalog)
    lat = np.repeat(catalog["latitude"].to_numpy(float), k)
    lon = np.repeat(catalog["longitude"].to_numpy(float), k)
    times = random_times(n, k, catalog["time"].min(), catalog["time"].max(), rng)
    total = n * k

    acc = _NullAccumulator()
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        s = vf.sample(times[start:end], lat[start:end], lon[start:end])
        acc.add(vf.compute_from_sample(s))
    return acc


# -- tests --------------------------------------------------------------------

def _cramers_v(chi2, n, cats):
    return float(np.sqrt(max(chi2, 0.0) / (n * max(cats - 1, 1))))


def _rows_for(obs: FeatureSet, acc: _NullAccumulator, stratum: str) -> list:
    rows = []
    for name, meta in obs.cat_meta.items():
        counts, n = obs.categorical_counts(name)
        p0 = acc.cat_prob(name)
        exp = p0 * n
        mask = exp > 0
        chi2 = float(np.sum((counts[mask] - exp[mask]) ** 2 / exp[mask]))
        dof = int(mask.sum()) - 1
        p = float(stats.chi2.sf(chi2, dof)) if dof > 0 else 1.0
        resid = np.where(exp > 0, (counts - exp) / np.sqrt(exp), 0.0)
        top = int(np.argmax(resid))
        rows.append(dict(stratum=stratum, family=meta.family, test=name, kind="chi2",
                         n=int(n), effect=_cramers_v(chi2, n, meta.n), p=p,
                         detail=f"top {meta.names[top]}: obs {counts[top]:.0f} vs exp {exp[top]:.1f} "
                                f"(z={resid[top]:+.2f})"))
    for name, family in obs.flag_family.items():
        cnt, n = obs.flag_count(name)
        p0 = acc.flag_prob(name)
        if not (0.0 < p0 < 1.0) or n == 0:
            continue
        pv = float(stats.binomtest(cnt, n, p0, alternative="two-sided").pvalue)
        ratio = (cnt / n) / p0 if p0 > 0 else np.nan
        rows.append(dict(stratum=stratum, family=family, test=name, kind="binom",
                         n=int(n), effect=float(ratio), p=pv,
                         detail=f"obs {cnt}/{n}={cnt/n:.4f} vs null {p0:.4f} (ratio {ratio:.2f})"))
    return rows


def analyze(catalog, vf: VedicFeatures, k, seed, stratum) -> list:
    obs = vf.compute(catalog["time"], catalog["latitude"], catalog["longitude"])
    acc = pooled_null(vf, catalog, k, seed)
    return _rows_for(obs, acc, stratum)


# -- CLI ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generic sidereal Vedic-astrology battery.")
    ap.add_argument("--kernel", default=os.environ.get("ASTRO_KERNEL", "de421.bsp"))
    ap.add_argument("--start", default="1973-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--minmag", type=float, default=6.0)
    ap.add_argument("--k", type=int, default=100, help="null replicates per event")
    ap.add_argument("--include", nargs="*", default=None,
                    help="only these vedic modules (e.g. panchanga declination)")
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    print(f"[1/4] Catalog M>={args.minmag} {args.start}..{args.end} ...")
    cat = fetch_catalog(args.start, args.end, args.minmag)
    main_all = decluster(cat)
    shallow = main_all[main_all["depth"] <= _SHALLOW_KM].reset_index(drop=True)
    print(f"      {len(cat)} events -> {len(main_all)} mainshocks ({len(shallow)} shallow)")

    print(f"[2/4] Loading ephemeris + battery (k={args.k}) ...")
    vf = VedicFeatures(args.kernel, include=args.include, exclude=args.exclude)
    print(f"      modules: {vf.modules}")

    rows = analyze(main_all, vf, args.k, args.seed, "all depths")
    rows += analyze(shallow, vf, args.k, args.seed + 1, f"shallow<={_SHALLOW_KM:g}km")

    print("[3/4] FDR-correcting ...")
    res = pd.DataFrame(rows)
    res["q_value"] = benjamini_hochberg(res["p"].to_numpy())
    res["sig_q<0.05"] = res["q_value"] < 0.05
    res = res.sort_values("p").reset_index(drop=True)
    _OUT.mkdir(parents=True, exist_ok=True)
    out_csv = _OUT / "vedic_results.csv"
    res.to_csv(out_csv, index=False)

    n_sig = int(res["sig_q<0.05"].sum())
    print(f"[4/4] {len(res)} tests across {res['family'].nunique()} families -> {out_csv}")
    pd.set_option("display.width", 180, "display.max_colwidth", 60)
    print(f"\n=== TOP 20 of {len(res)} tests (BH-FDR); {n_sig} significant at q<0.05 ===")
    print(res.head(20).to_string(index=False))

    sun = res[res["test"] == "sign_Sun"]
    if not sun.empty:
        print("\nNegative control (sign_Sun = season): "
              f"min p={sun['p'].min():.3f}, min q={sun['q_value'].min():.3f}")


if __name__ == "__main__":
    main()
