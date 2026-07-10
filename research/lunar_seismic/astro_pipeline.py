"""Exploratory sidereal ("astrology") battery vs a random-time null.

Tests whether the sidereal sky at earthquakes departs from what random timing
would produce, across a large family of features:

    * zodiac sign of each of the 9 grahas         (chi-square, 12 cats)
    * ascendant sign at the epicenter             (chi-square, 12 cats)
    * Moon's nakshatra                            (chi-square, 27 cats)
    * retrograde state of the 5 star-planets      (binomial)
    * graha-graha aspects: conjunction / opposition / square within an orb
                                                  (binomial, all pairs)

Every p-value is corrected together with Benjamini-Hochberg FDR. This is an
exploratory sweep: a hit means "worth a pre-registered, out-of-sample retest",
not a proven effect -- and certainly not causation. The Sun's sign (= time of
year) is included as a negative control.
"""
from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .catalog import fetch_catalog
from .decluster import decluster
from .nulls import random_times
from .stats import benjamini_hochberg
from .astro_features import (
    AstroFeatures, ALL_GRAHAS, STAR_PLANETS, SIGN_NAMES, NAKSHATRA_NAMES,
)

_OUT = Path(__file__).resolve().parent / "outputs"
_SHALLOW_KM = 70.0
ASPECTS = {"conjunction": 0.0, "opposition": 180.0, "square": 90.0}
DEFAULT_ORB = 6.0
_GRAHA_NAMES = [g.value for g in ALL_GRAHAS]


def aspect_pairs():
    pairs = list(combinations(_GRAHA_NAMES, 2))
    # Rahu-Ketu are 180 deg apart by construction: drop the degenerate pair.
    return [(a, b) for (a, b) in pairs if {a, b} != {"Rahu", "Ketu"}]


def _sep(l1, l2):
    d = np.abs(l1 - l2) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def _aspect_present(lon, a, b, angle, orb):
    s = _sep(lon[a], lon[b])
    return s <= orb if angle == 0.0 else np.abs(s - angle) <= orb


def _observed_counts(feats, pairs, orb):
    n = len(feats["moon_nakshatra"])
    obs = {}
    for g in _GRAHA_NAMES:
        obs[("sign", g)] = np.bincount(feats["_sign"][g], minlength=12).astype(float)
    obs[("asc",)] = np.bincount(feats["asc_sign"], minlength=12).astype(float)
    obs[("nak",)] = np.bincount(feats["moon_nakshatra"], minlength=27).astype(float)
    for p in STAR_PLANETS:
        obs[("retro", p.value)] = float(feats["_retro"][p.value].sum())
    for (a, b) in pairs:
        for asp, angle in ASPECTS.items():
            obs[("aspect", a, b, asp)] = float(_aspect_present(feats["_lon"], a, b, angle, orb).sum())
    return obs, n


def _blank_like(obs):
    acc = {}
    for key, val in obs.items():
        acc[key] = np.zeros_like(val) if isinstance(val, np.ndarray) else 0.0
    return acc


def pooled_null(af, catalog, k, seed, pairs, orb, chunk=50_000):
    """Pooled category probabilities under random-time null (matched epicenters)."""
    rng = np.random.default_rng(seed)
    n = len(catalog)
    lat = np.repeat(catalog["latitude"].to_numpy(float), k)
    lon = np.repeat(catalog["longitude"].to_numpy(float), k)
    times = random_times(n, k, catalog["time"].min(), catalog["time"].max(), rng)
    total = n * k

    acc = None
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        f = af.compute(times[start:end], lat[start:end], lon[start:end])
        oc, _ = _observed_counts(f, pairs, orb)
        if acc is None:
            acc = _blank_like(oc)
        for key, val in oc.items():
            acc[key] = acc[key] + val
    return {key: (val / total) for key, val in acc.items()}, total


def _cramers_v(chi2, n, cats):
    return float(np.sqrt(max(chi2, 0.0) / (n * (cats - 1))))


def _test_rows(observed, n, null_prob, stratum):
    rows = []
    # Categorical chi-square tests (sign / ascendant / nakshatra).
    cat_specs = [(("sign", g), 12, SIGN_NAMES, f"{g} sign") for g in _GRAHA_NAMES]
    cat_specs.append((("asc",), 12, SIGN_NAMES, "Ascendant sign"))
    cat_specs.append((("nak",), 27, NAKSHATRA_NAMES, "Moon nakshatra"))
    for key, ncat, names, label in cat_specs:
        obs = observed[key]
        exp = null_prob[key] * n
        mask = exp > 0
        chi2 = float(np.sum((obs[mask] - exp[mask]) ** 2 / exp[mask]))
        dof = int(mask.sum()) - 1
        p = float(stats.chi2.sf(chi2, dof)) if dof > 0 else 1.0
        resid = np.where(exp > 0, (obs - exp) / np.sqrt(exp), 0.0)
        top = int(np.argmax(resid))
        rows.append(dict(stratum=stratum, family="sign/house", test=label, n=int(n),
                         effect=_cramers_v(chi2, n, ncat), p=p,
                         detail=f"most enriched: {names[top]} obs {obs[top]:.0f} vs exp {exp[top]:.1f}"))
    # Binomial: retrograde.
    for p_ in STAR_PLANETS:
        key = ("retro", p_.value)
        cnt = int(observed[key]); p0 = float(null_prob[key])
        pv = stats.binomtest(cnt, n, p0, alternative="two-sided").pvalue if 0 < p0 < 1 else 1.0
        ratio = (cnt / n) / p0 if p0 > 0 else np.nan
        rows.append(dict(stratum=stratum, family="retrograde", test=f"{p_.value} retrograde",
                         n=int(n), effect=ratio, p=float(pv),
                         detail=f"obs {cnt}/{n}={cnt/n:.3f} vs null {p0:.3f}"))
    # Binomial: aspects.
    for key, cnt in observed.items():
        if key[0] != "aspect":
            continue
        _, a, b, asp = key
        p0 = float(null_prob[key])
        if p0 <= 0:
            continue
        cnt = int(cnt)
        pv = stats.binomtest(cnt, n, p0, alternative="two-sided").pvalue
        ratio = (cnt / n) / p0
        rows.append(dict(stratum=stratum, family="aspect", test=f"{a}-{b} {asp}",
                         n=int(n), effect=ratio, p=float(pv),
                         detail=f"obs {cnt}/{n}={cnt/n:.3f} vs null {p0:.3f} (ratio {ratio:.2f})"))
    return rows


def analyze(catalog, af, k, seed, stratum, pairs, orb):
    feats = af.compute(catalog["time"], catalog["latitude"], catalog["longitude"])
    observed, n = _observed_counts(feats, pairs, orb)
    null_prob, _ = pooled_null(af, catalog, k, seed, pairs, orb)
    return _test_rows(observed, n, null_prob, stratum)


def main():
    ap = argparse.ArgumentParser(description="Exploratory sidereal astrology battery.")
    ap.add_argument("--kernel", default=os.environ.get("ASTRO_KERNEL", "de421.bsp"))
    ap.add_argument("--start", default="1973-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--minmag", type=float, default=6.0)
    ap.add_argument("--k", type=int, default=100, help="null replicates per event")
    ap.add_argument("--orb", type=float, default=DEFAULT_ORB)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    print(f"[1/4] Catalog M>={args.minmag} {args.start}..{args.end} ...")
    cat = fetch_catalog(args.start, args.end, args.minmag)
    main_all = decluster(cat)
    shallow = main_all[main_all["depth"] <= _SHALLOW_KM].reset_index(drop=True)
    print(f"      {len(cat)} events -> {len(main_all)} mainshocks ({len(shallow)} shallow)")

    print(f"[2/4] Loading ephemeris + computing battery (k={args.k}, orb={args.orb} deg) ...")
    af = AstroFeatures(args.kernel)
    pairs = aspect_pairs()

    rows = analyze(main_all, af, args.k, args.seed, "all depths", pairs, args.orb)
    rows += analyze(shallow, af, args.k, args.seed + 1, f"shallow <= {_SHALLOW_KM:g} km", pairs, args.orb)

    print("[3/4] FDR-correcting ...")
    res = pd.DataFrame(rows)
    res["q_value"] = benjamini_hochberg(res["p"].to_numpy())
    res["sig_q<0.05"] = res["q_value"] < 0.05
    res = res.sort_values("p").reset_index(drop=True)
    _OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(_OUT / "astro_results.csv", index=False)

    print(f"[4/4] {len(res)} tests. Writing {_OUT/'astro_results.csv'}")
    pd.set_option("display.width", 170, "display.max_colwidth", 52)
    n_sig = int(res["sig_q<0.05"].sum())
    print(f"\n=== TOP 15 of {len(res)} tests (BH-FDR); {n_sig} significant at q<0.05 ===")
    print(res.head(15).to_string(index=False))
    sun = res[res["test"] == "Sun sign"]
    if not sun.empty:
        print("\nNegative control (Sun sign = season): "
              f"p={sun.iloc[0]['p']:.3f}, q={sun.iloc[0]['q_value']:.3f}")


if __name__ == "__main__":
    main()
