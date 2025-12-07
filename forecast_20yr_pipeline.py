#!/usr/bin/env python3
"""
forecast_20yr_pipeline.py (mapping-fix patch)

Patch summary
- Fixes a bug where GLM fallback forecasts were reported for historical training years
  because the code reused the training-design row mapping when assembling forecast counts.
- For GLM path we now construct a proper forecast design matrix (X_forecast) for the
  requested forecast years and clusters, using the same column order as the GLM design
  used for fitting. We then compute lam_matrix = exp(draws @ X_forecast.T) to produce
  counts for the requested forecast years instead of reusing training rows.

Other behavior is unchanged (vectorized sampling, optional PyMC, optional expand-events).
Run with --no-pymc to force GLM fallback if your env cannot compile PyMC/pytensor.

Usage (example):
  python forecast_20yr_pipeline.py --input updated.csv --start-year 2026 --horizon-years 20 \
    --time-bin yearly --ensembles 500 --output-prefix forecast_2026_2045 --no-pymc
"""
from __future__ import annotations
import argparse
import math
import os
from datetime import datetime
import numpy as np
import pandas as pd
import random
from dateutil import parser as dtparser

# optional libraries
try:
    import hdbscan
except Exception:
    hdbscan = None
try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None
try:
    import pymc as pm
    import arviz as az
except Exception:
    pm = None
    az = None
try:
    import statsmodels.api as sm
except Exception:
    sm = None

# -----------------------
# CLI / defaults
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--time-col", default="time")
    p.add_argument("--lat-col", default="latitude")
    p.add_argument("--lon-col", default="longitude")
    p.add_argument("--mag-col", default="mag")
    p.add_argument("--start-year", type=int, default=2026)
    p.add_argument("--horizon-years", type=int, default=20)
    p.add_argument("--time-bin", choices=["yearly","monthly"], default="yearly")
    p.add_argument("--ensembles", type=int, default=100, help="Number of ensemble realizations (default 100)")
    p.add_argument("--output-prefix", default="forecast_2026_2045")
    p.add_argument("--clusters-min-samples", type=int, default=5, help="HDBSCAN min_samples")
    p.add_argument("--min-cluster-size", type=int, default=25, help="HDBSCAN min_cluster_size")
    p.add_argument("--min-cluster-count", type=int, default=10, help="Merge clusters with total historical count < this into 'other' to reduce dimensionality")
    p.add_argument("--use-neg-bin", action="store_true", help="fit Negative Binomial instead of Poisson (via overdispersion)")
    p.add_argument("--dbscan-eps-km", type=float, default=50.0, help="DBSCAN eps (km) when hdbscan unavailable")
    p.add_argument("--no-pymc", action="store_true", help="Force using statsmodels GLM fallback even if PyMC is installed")
    p.add_argument("--draws", type=int, default=500, help="PyMC draws (if used)")
    p.add_argument("--tune", type=int, default=500, help="PyMC tune (if used)")
    p.add_argument("--expand-events", action="store_true", help="Expand aggregated counts into per-event sampled rows (can be very large)")
    return p.parse_args()

# -----------------------
# Utilities
# -----------------------
def ensure_datetime_series(df, col):
    s = pd.to_datetime(df[col], utc=True, errors='coerce')
    if s.isna().any():
        s2 = df[col].astype(str).apply(lambda x: dtparser.parse(x) if x and not pd.isna(x) else pd.NaT)
        s = pd.to_datetime(s2, utc=True, errors='coerce')
    return s

# -----------------------
# Clustering with small-cluster merging
# -----------------------
def compute_spatial_clusters(df, lat_col="latitude", lon_col="longitude",
                             min_cluster_size=25, min_samples=5, dbscan_eps_km=50.0, min_cluster_count=10):
    coords = df[[lat_col, lon_col]].to_numpy(dtype=float)
    if np.isnan(coords).any():
        mask_valid = ~np.isnan(coords).any(axis=1)
        coords_valid = coords[mask_valid]
    else:
        mask_valid = np.ones(len(coords), dtype=bool)
        coords_valid = coords

    if hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
        labels_valid = clusterer.fit_predict(coords_valid)
        labels = np.full(len(coords), -1, dtype=int)
        labels[mask_valid] = labels_valid
        df = df.copy()
        df["cluster"] = labels
    else:
        if DBSCAN is None:
            raise RuntimeError("Neither hdbscan nor sklearn DBSCAN is available.")
        coords_rad = np.radians(coords_valid)
        eps_rad = float(dbscan_eps_km) / 6371.0
        db = DBSCAN(eps=eps_rad, min_samples=max(1, min_samples), metric='haversine', n_jobs=-1)
        labels_valid = db.fit_predict(coords_rad)
        labels = np.full(len(coords), -1, dtype=int)
        labels[mask_valid] = labels_valid
        df = df.copy()
        df["cluster"] = labels

    # Merge small clusters into -1 to reduce dimensionality if requested
    counts = df["cluster"].value_counts()
    small = counts[counts < min_cluster_count].index.tolist()
    if small:
        mask_small = df["cluster"].isin(small)
        df.loc[mask_small, "cluster"] = -1
    return df

# -----------------------
# Aggregation
# -----------------------
def aggregate_counts(df, time_col="time", cluster_col="cluster", bin="yearly"):
    if bin == "yearly":
        df["year"] = pd.to_datetime(df[time_col]).dt.year
        agg = df.groupby([cluster_col, "year"]).size().rename("count").reset_index()
    else:
        df["month"] = pd.to_datetime(df[time_col]).dt.to_period("M").dt.to_timestamp()
        agg = df.groupby([cluster_col, "month"]).size().rename("count").reset_index()
    return agg

# -----------------------
# Fit hierarchical model (PyMC optional) or GLM fallback (statsmodels)
# -----------------------
def fit_hierarchical_model(agg, time_bin="yearly", use_neg_bin=False, draws=500, tune=500, allow_pymc=True):
    # prepare indices
    if time_bin == "yearly":
        time_idx = agg["year"].astype(int)
    else:
        time_idx = (agg["month"].dt.year - agg["month"].dt.year.min())*12 + (agg["month"].dt.month-1)
    clusters_unique, cluster_idx = np.unique(agg["cluster"], return_inverse=True)
    t0 = time_idx.min()
    time_rel = (time_idx - t0).astype(int)
    K = len(clusters_unique)

    # Try PyMC if allowed and installed
    if allow_pymc and pm is not None:
        with pm.Model() as model:
            mu_a = pm.Normal("mu_a", mu=0.0, sigma=5.0)
            sigma_a = pm.Exponential("sigma_a", 1.0)
            a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=K)
            beta_time = pm.Normal("beta_time", mu=0.0, sigma=1.0)
            log_lambda = a[cluster_idx] + beta_time * time_rel
            lambda_ = pm.math.exp(log_lambda)
            if use_neg_bin:
                alpha = pm.Exponential("alpha", 1.0)
                obs = pm.NegativeBinomial("obs", mu=lambda_, alpha=alpha, observed=agg["count"].values)
            else:
                obs = pm.Poisson("obs", mu=lambda_, observed=agg["count"].values)
            trace = pm.sample(draws=draws, tune=tune, cores=1, progressbar=True)
        return ("pymc", model, trace, clusters_unique, t0)

    # Fallback to GLM (statsmodels) and return result info
    if sm is None:
        raise RuntimeError("statsmodels is required for GLM fallback. Install statsmodels or disable --no-pymc accordingly.")

    df_design = agg.copy()
    df_design["time_rel"] = time_rel
    dummies = pd.get_dummies(df_design["cluster"].astype(str), prefix="cl", drop_first=True)
    X = pd.concat([pd.Series(1, index=df_design.index, name="intercept"), dummies, df_design["time_rel"].rename("time_rel")], axis=1)
    y = df_design["count"]
    # Coerce numeric and convert to numpy arrays for speed
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    y = pd.to_numeric(y, errors='coerce').fillna(0.0).astype(float)
    family = sm.families.NegativeBinomial() if use_neg_bin else sm.families.Poisson()
    glm_model = sm.GLM(y, X, family=family)
    glm_res = glm_model.fit()
    result_info = {
        "params": glm_res.params.astype(float),
        "cov": glm_res.cov_params().astype(float),
        "design_columns": X.columns.tolist(),
        "X": X,        # keep X matrix for reference
        "clusters_unique": clusters_unique,
        "t0": t0
    }
    return ("glm", glm_model, result_info, clusters_unique, t0)

# -----------------------
# Vectorized ensemble sampling (fixed GLM mapping)
# -----------------------
def sample_ensembles_vectorized(model_tuple, start_year, horizon_years, ensembles=100, use_neg_bin=False, rng_seed=12345):
    kind = model_tuple[0]
    years = np.arange(start_year, start_year + horizon_years)

    if kind == "pymc":
        _, model, trace, clusters, t0 = model_tuple
        posterior = trace.posterior
        a_all = posterior["a"].values.reshape(-1, posterior["a"].shape[-1])  # (n_samples, K)
        beta_all = posterior["beta_time"].values.reshape(-1)
        n_posterior = a_all.shape[0]
        rng = np.random.default_rng(rng_seed)
        idxs = rng.integers(0, n_posterior, size=ensembles)
        a_draws = a_all[idxs]         # (ensembles, K)
        beta_draws = beta_all[idxs]   # (ensembles,)
        design_rows = []
        for c_idx, c in enumerate(clusters):
            for y in years:
                time_rel = y - t0
                design_rows.append((c_idx, int(c), int(y), int(time_rel)))
        design_df = pd.DataFrame(design_rows, columns=["cluster_idx","cluster","year","time_rel"])
        time_rel_arr = design_df["time_rel"].to_numpy()
        cluster_idx_arr = design_df["cluster_idx"].to_numpy()
        lam = np.exp(a_draws[:, cluster_idx_arr] + (beta_draws[:, None] * time_rel_arr[None, :]))
        rng = np.random.default_rng(rng_seed + 1)
        counts = rng.poisson(lam)
        ensembles_list = []
        for e in range(ensembles):
            df_e = pd.DataFrame({
                "ensemble": e,
                "cluster": design_df["cluster"].to_numpy(),
                "year": design_df["year"].to_numpy(),
                "count": counts[e, :]
            })
            ensembles_list.append(df_e)
        ensembles_counts = pd.concat(ensembles_list, ignore_index=True)
        return ensembles_counts

    # GLM path (fixed): construct forecast design matrix (X_forecast) that matches GLM design_columns
    if kind == "glm":
        _, glm_model, result_info, clusters_unique, t0 = model_tuple
        X_fit = result_info["X"]                     # training design
        cols = result_info["design_columns"]         # column order used in fit
        mean_vec = result_info["params"].reindex(cols).fillna(0.0).values
        cov_mat = result_info["cov"].reindex(index=cols, columns=cols).fillna(0.0).values
        rng = np.random.default_rng(rng_seed)
        draws = rng.multivariate_normal(mean_vec, cov_mat, size=ensembles)  # (ensembles, p)

        # Build forecast design rows for all clusters and forecast years
        design_rows = []
        cluster_list = list(clusters_unique)
        for c_idx, c in enumerate(cluster_list):
            for y in years:
                time_rel = y - t0
                design_rows.append({"cluster": c, "year": int(y), "time_rel": time_rel})
        design_df = pd.DataFrame(design_rows)

        # Create dummies consistent with training (drop_first=True behavior)
        # We'll create dummies with same prefix and then reindex columns to match 'cols'
        design_df["cluster_str"] = design_df["cluster"].astype(str)
        dummies_fore = pd.get_dummies(design_df["cluster_str"], prefix="cl", drop_first=True)

        # Build X_forecast DataFrame with same columns as 'cols'
        X_fore = pd.DataFrame(index=design_df.index)
        X_fore["intercept"] = 1.0
        X_fore["time_rel"] = design_df["time_rel"].astype(float)
        # add dummy columns (may be subset of cols)
        for ccol in [c for c in cols if c.startswith("cl_")]:
            if ccol in dummies_fore.columns:
                X_fore[ccol] = dummies_fore[ccol].astype(float)
            else:
                X_fore[ccol] = 0.0
        # ensure column order matches 'cols'
        X_fore = X_fore.reindex(columns=cols, fill_value=0.0).astype(float)

        # compute lam_matrix = exp( draws @ X_fore.T )
        X_T = X_fore.values.T   # (p, n_rows)
        lam_matrix = np.exp( draws.dot(X_T) )   # (ensembles, n_rows)

        rng2 = np.random.default_rng(rng_seed + 1)
        counts = rng2.poisson(lam_matrix)   # (ensembles, n_rows)

        # Assemble aggregated DataFrame mapping design_df rows to counts columns
        records = []
        n_rows = lam_matrix.shape[1]
        for e in range(ensembles):
            rec = pd.DataFrame({
                "ensemble": e,
                "cluster": design_df["cluster"].to_numpy(),
                "year": design_df["year"].to_numpy(),
                "count": counts[e, :]
            })
            records.append(rec)
        ensembles_counts = pd.concat(records, ignore_index=True)
        return ensembles_counts

    raise RuntimeError("Unknown model kind")

# -----------------------
# Efficient expansion to per-event sampled rows (vectorized)
# -----------------------
def expand_aggregated_counts_to_events(aggregated_df, original_df, lat_col="latitude", lon_col="longitude", mag_col="mag"):
    rng = np.random.default_rng(2023)
    cluster_groups = {}
    for c, grp in original_df.groupby("cluster"):
        cluster_groups[c] = {
            "lat": grp[lat_col].to_numpy(),
            "lon": grp[lon_col].to_numpy(),
            "mag": grp[mag_col].to_numpy(),
            "n": len(grp)
        }
    events_rows = []
    for _, row in aggregated_df.iterrows():
        cnt = int(row["count"])
        if cnt <= 0:
            continue
        c = row["cluster"]
        year = int(row["year"])
        ensemble = int(row["ensemble"])
        grp = cluster_groups.get(c, None)
        if grp is None or grp["n"] == 0:
            lat = np.random.uniform(original_df[lat_col].min(), original_df[lat_col].max(), size=cnt)
            lon = np.random.uniform(original_df[lon_col].min(), original_df[lon_col].max(), size=cnt)
            mag = np.random.normal(original_df[mag_col].mean(), original_df[mag_col].std(), size=cnt)
        else:
            idxs = rng.integers(0, grp["n"], size=cnt)
            lat = grp["lat"][idxs]
            lon = grp["lon"][idxs]
            mag = grp["mag"][idxs]
        days = rng.integers(1, 366, size=cnt)
        times = [ (datetime(year,1,1) + pd.to_timedelta(int(d-1), unit='D')).isoformat() for d in days ]
        events_rows.extend([{"ensemble": int(ensemble), "cluster": int(c), "time": t, "year": year, "latitude": float(lat_v), "longitude": float(lon_v), "mag": float(m_v)} for t, lat_v, lon_v, m_v in zip(times, lat, lon, mag)])
    if not events_rows:
        return pd.DataFrame(columns=["ensemble","cluster","time","year","latitude","longitude","mag"])
    return pd.DataFrame(events_rows)

# -----------------------
# Main workflow
# -----------------------
def main():
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    df[args.time_col] = ensure_datetime_series(df, args.time_col)
    for col in [args.time_col, args.lat_col, args.lon_col, args.mag_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    print("Clustering spatially (HDBSCAN preferred, DBSCAN fallback)...")
    df = compute_spatial_clusters(df, lat_col=args.lat_col, lon_col=args.lon_col,
                                  min_cluster_size=args.min_cluster_size, min_samples=args.clusters_min_samples,
                                  dbscan_eps_km=args.dbscan_eps_km, min_cluster_count=args.min_cluster_count)
    print("Cluster counts (top 10):")
    print(df["cluster"].value_counts().head(10))

    agg = aggregate_counts(df, time_col=args.time_col, cluster_col="cluster", bin=args.time_bin)
    print("Fitting hierarchical model (PyMC if available and not disabled, otherwise statsmodels GLM)...")
    allow_pymc = (not args.no_pymc)
    model_tuple = fit_hierarchical_model(agg, time_bin=args.time_bin, use_neg_bin=args.use_neg_bin, draws=args.draws, tune=args.tune, allow_pymc=allow_pymc)

    print("Sampling ensembles (vectorized)...")
    sampled = sample_ensembles_vectorized(model_tuple, start_year=args.start_year, horizon_years=args.horizon_years, ensembles=args.ensembles, use_neg_bin=args.use_neg_bin)

    ensembles_counts = sampled  # sample_ensembles_vectorized now returns aggregated DF directly for both paths

    out_prefix = args.output_prefix
    counts_path = f"{out_prefix}_counts.parquet"
    ensembles_counts.to_parquet(counts_path, index=False)
    print("Wrote aggregated ensemble counts to:", counts_path)

    summary = ensembles_counts.groupby(["ensemble","year"])["count"].sum().reset_index(name="count")
    summary_q = summary.groupby("year")["count"].agg([("p5", lambda x: np.quantile(x,0.05)),
                                                      ("p50", lambda x: np.quantile(x,0.5)),
                                                      ("p95", lambda x: np.quantile(x,0.95)),
                                                      ("mean", "mean")]).reset_index()
    summary_path = f"{out_prefix}_summary.csv"
    summary_q.to_csv(summary_path, index=False)
    print("Wrote per-year summary to:", summary_path)

    if args.expand_events:
        print("Expanding aggregated counts to per-event sampled rows (may be slow/large)...")
        events_df = expand_aggregated_counts_to_events(ensembles_counts, df, lat_col=args.lat_col, lon_col=args.lon_col, mag_col=args.mag_col)
        events_path = f"{out_prefix}_events.parquet"
        events_df.to_parquet(events_path, index=False)
        print("Wrote per-event ensembles to:", events_path)

    print("Done.")

if __name__ == "__main__":
    main()