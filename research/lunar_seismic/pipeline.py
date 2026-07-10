"""End-to-end lunar tidal-triggering analysis.

Run from the repo root, with the DE-ephemeris kernel available:

    python -m research.lunar_seismic.pipeline \
        --kernel <path-to-de421.bsp> --minmag 6.0 --k 200

Steps: fetch catalog -> decluster to independent mainshocks -> compute each
event's lunar tidal geometry -> build a random-time null -> run Schuster and
Monte-Carlo tests (all events and shallow-only) -> FDR-correct -> write a results
table and diagnostic plots to ``outputs/``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import fetch_catalog
from .decluster import decluster
from .geometry import TidalGeometry, FEATURE_COLUMNS
from .nulls import null_features
from .stats import schuster_test, mc_test, benjamini_hochberg

_OUT = Path(__file__).resolve().parent / "outputs"
_FIDX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
_SHALLOW_KM = 70.0


def _derive(observed: pd.DataFrame, null: np.ndarray):
    """Add derived features (|dec|, syzygy index) to observed frame + null array.

    syzygy_index = cos(2*elongation): +1 at new/full (spring tide), -1 at
    quadrature (neap tide).
    """
    obs = observed.copy()
    obs["abs_moon_dec"] = obs["moon_dec_deg"].abs()
    obs["syzygy_index"] = np.cos(2 * np.radians(obs["sun_moon_elong_deg"]))

    dec = null[:, :, _FIDX["moon_dec_deg"]]
    elong = null[:, :, _FIDX["sun_moon_elong_deg"]]
    null_abs_dec = np.abs(dec)
    null_syzygy = np.cos(2 * np.radians(elong))
    return obs, {"abs_moon_dec": null_abs_dec, "syzygy_index": null_syzygy}


def _null_col(null: np.ndarray, derived_null: dict, feature: str) -> np.ndarray:
    if feature in _FIDX:
        return null[:, :, _FIDX[feature]]
    return derived_null[feature]


# (feature, direction, human description). direction: 'greater'/'less' one-sided.
_MC_TESTS = [
    ("tide_vertical", "greater", "Higher combined vertical tide (Moon+Sun overhead/underfoot)"),
    ("abs_moon_dec", "greater", "Larger |lunar declination| (near standstill)"),
    ("moon_dist_km", "less", "Smaller Earth-Moon distance (perigee / stronger tide)"),
    ("syzygy_index", "greater", "Nearer syzygy (spring tide) vs quadrature"),
    ("tide_total_gm_d3", "greater", "Higher raw tidal strength GM/d^3 (perigee+syzygy)"),
]


def analyze(mainshocks: pd.DataFrame, geom: TidalGeometry, k: int, seed: int, stratum: str):
    obs = geom.features(mainshocks["time"], mainshocks["latitude"], mainshocks["longitude"])
    null = null_features(geom, mainshocks, k=k, seed=seed)
    obs, derived_null = _derive(obs, null)

    rows = []

    # Schuster tests on the lunar hour-angle phase (0 deg = Moon overhead).
    ha_deg = obs["moon_hour_angle_h"].to_numpy() * 15.0
    diurnal = schuster_test(ha_deg)
    semidiurnal = schuster_test((ha_deg * 2.0) % 360.0)
    rows.append(dict(stratum=stratum, test="Schuster diurnal (hour angle)",
                     stat=diurnal["rbar"], p=diurnal["p"], effect=diurnal["rbar"],
                     detail=f"preferred hour angle {diurnal['phase']/15.0:.2f} h", n=diurnal["n"]))
    rows.append(dict(stratum=stratum, test="Schuster semidiurnal (Moon at top/bottom vs horizon)",
                     stat=semidiurnal["rbar"], p=semidiurnal["p"], effect=semidiurnal["rbar"],
                     detail=f"preferred semidiurnal phase {semidiurnal['phase']:.1f} deg", n=semidiurnal["n"]))

    # Monte-Carlo tests on continuous tidal features.
    for feat, direction, desc in _MC_TESTS:
        res = mc_test(obs[feat].to_numpy(), _null_col(null, derived_null, feat))
        p = res.p_greater if direction == "greater" else res.p_less
        rows.append(dict(stratum=stratum, test=desc, stat=res.obs_mean, p=p,
                         effect=res.z, detail=f"null mean {res.null_mean:.4g}, z={res.z:+.2f}",
                         n=res.n))
    return obs, null, rows


def make_plots(obs_all: pd.DataFrame, null_all: np.ndarray, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _OUT.mkdir(parents=True, exist_ok=True)

    # 1. Lunar hour-angle histogram: observed vs null expectation.
    ha = obs_all["moon_hour_angle_h"].to_numpy()
    null_ha = null_all[:, :, _FIDX["moon_hour_angle_h"]].ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-12, 12, 25)
    ax.hist(ha, bins=bins, density=True, alpha=0.6, label="earthquakes")
    ax.hist(null_ha, bins=bins, density=True, histtype="step", color="k", label="random-time null")
    ax.set_xlabel("Lunar hour angle (h): 0=overhead, ±6=horizon, ±12=underfoot")
    ax.set_ylabel("density"); ax.set_title(f"Lunar hour angle at earthquakes ({tag})"); ax.legend()
    fig.tight_layout(); fig.savefig(_OUT / f"hour_angle_{tag}.png", dpi=130); plt.close(fig)

    # 2. Vertical-tide: observed distribution vs null.
    tv = obs_all["tide_vertical"].to_numpy()
    null_tv = null_all[:, :, _FIDX["tide_vertical"]].ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_tv, bins=40, density=True, histtype="step", color="k", label="random-time null")
    ax.hist(tv, bins=40, density=True, alpha=0.6, label="earthquakes")
    ax.axvline(tv.mean(), color="C0", ls="--"); ax.axvline(null_tv.mean(), color="k", ls=":")
    ax.set_xlabel("combined vertical tidal acceleration (relative units)")
    ax.set_ylabel("density"); ax.set_title(f"Vertical tide at earthquakes ({tag})"); ax.legend()
    fig.tight_layout(); fig.savefig(_OUT / f"tide_vertical_{tag}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Lunar tidal-triggering analysis.")
    ap.add_argument("--kernel", default=os.environ.get("ASTRO_KERNEL", "de421.bsp"))
    ap.add_argument("--start", default="1973-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--minmag", type=float, default=6.0)
    ap.add_argument("--k", type=int, default=200, help="null replicates per event")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print(f"[1/5] Fetching catalog M>={args.minmag} {args.start}..{args.end} ...")
    cat = fetch_catalog(args.start, args.end, args.minmag)
    print(f"      {len(cat)} events")

    print("[2/5] Declustering (Gardner-Knopoff) ...")
    main_all = decluster(cat)
    shallow = main_all[main_all["depth"] <= _SHALLOW_KM].reset_index(drop=True)
    print(f"      {len(main_all)} mainshocks ({len(shallow)} shallow <= {_SHALLOW_KM:g} km)")

    print(f"[3/5] Loading ephemeris {args.kernel} + computing geometry & null (k={args.k}) ...")
    geom = TidalGeometry(args.kernel)

    all_rows = []
    obs_all, null_all, rows = analyze(main_all, geom, args.k, args.seed, "all depths")
    all_rows += rows
    _, _, rows_sh = analyze(shallow, geom, args.k, args.seed + 1, f"shallow <= {_SHALLOW_KM:g} km")
    all_rows += rows_sh

    print("[4/5] FDR-correcting and writing results ...")
    res = pd.DataFrame(all_rows)
    res["q_value"] = benjamini_hochberg(res["p"].to_numpy())
    res["significant_q<0.05"] = res["q_value"] < 0.05
    res = res[["stratum", "test", "n", "stat", "effect", "p", "q_value",
               "significant_q<0.05", "detail"]]
    _OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(_OUT / "results.csv", index=False)

    print("[5/5] Plotting ...")
    make_plots(obs_all, null_all, "all_depths")

    pd.set_option("display.width", 160, "display.max_colwidth", 60)
    print("\n=== RESULTS (random-time null, Benjamini-Hochberg FDR) ===")
    print(res.to_string(index=False))
    print(f"\nWrote {_OUT/'results.csv'} and plots to {_OUT}")


if __name__ == "__main__":
    main()
