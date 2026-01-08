#!/usr/bin/env python3
"""
label_patterns.py

Cluster rows in events_celestial.csv into human-meaningful "patterns" using a
configurable feature set. Ensures latitude_num and longitude_num are available
and optionally applies a spatial weight multiplier to increase/decrease the
influence of geographic location.

Output:
 - A CSV with added columns: pattern_id (int), pattern_label (string),
   pattern_name (human-friendly), pattern_size (int), pattern_method (str),
   pattern_features (comma-separated features used)
 - Print a short summary to stdout.

Usage examples:
  python src/label_patterns.py --input events_celestial.csv --out events_celestial.patterned.csv
  python src/label_patterns.py --input events_celestial.csv --out events_celestial.patterned.csv \
      --features "_jd_,latitude_num,longitude_num,mag,Sun_sid_long,Moon_sid_long" \
      --method kmeans --n-clusters 12 --spatial-weight 2.0

Notes:
 - Numeric features are median-imputed; categorical features are ordinal-encoded.
 - Standard scaling is applied by default; can be disabled with --no-scale.
 - Spatial weight multiplies latitude_num/longitude_num prior to scaling.
"""
from __future__ import annotations
import argparse
import math
import sys
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


DEFAULT_FEATURE_CANDIDATES = [
    "_jd_", "latitude_num", "longitude_num", "mag", "depth",
    # apparent RA/Dec
    "Sun_ra_hours", "Moon_ra_hours",
    "Sun_dec_deg", "Moon_dec_deg",
    # sidereal longitudes
    "Sun_sid_long", "Moon_sid_long", "Mars_sid_long", "Saturn_sid_long", "Venus_sid_long",
    # altitudes
    "Sun_altitude_deg", "Moon_altitude_deg",
    # heliocentric
    "Sun_helio_lon_deg", "Moon_helio_lon_deg",
]


def detect_feature_columns(df: pd.DataFrame, requested: Optional[List[str]] = None) -> List[str]:
    """
    If requested list provided, return intersection with df columns (warn on missing).
    Otherwise auto-detect a default set including lat/lon numeric columns and any *_zodiac.
    """
    if requested:
        present = [c for c in requested if c in df.columns]
        missing = [c for c in requested if c not in df.columns]
        if missing:
            print(f"Warning: requested features missing and will be skipped: {missing}")
        return present

    features = [c for c in DEFAULT_FEATURE_CANDIDATES if c in df.columns]
    # include any zodiac columns (categorical)
    zodiac_cols = [c for c in df.columns if c.endswith("_zodiac")]
    features += zodiac_cols
    # ensure latitude_num/longitude_num included when possible
    for s in ("latitude_num", "longitude_num"):
        if s in df.columns and s not in features:
            features.insert(0, s)
    # unique preserve order
    seen = set()
    ordered = []
    for f in features:
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    return ordered


def prepare_matrix(df: pd.DataFrame, features: List[str], spatial_weight: float = 1.0) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """
    Build numeric matrix X from df using features list.
    - Numeric columns: median-imputed
    - Categorical columns: ordinal-encoded
    - spatial_weight multiplies latitude_num/longitude_num (before scaling)
    Returns: X, used_feature_list, feature_types_map
    """
    types: Dict[str, str] = {}
    final_feats: List[str] = []

    numeric_cols = []
    cat_cols = []
    for f in features:
        if f not in df.columns:
            continue
        ser = df[f]
        if pd.api.types.is_numeric_dtype(ser):
            numeric_cols.append(f)
            types[f] = "numeric"
            final_feats.append(f)
        else:
            cat_cols.append(f)
            types[f] = "categorical"
            final_feats.append(f)

    mat_parts = []
    # numeric handling
    if numeric_cols:
        num_df = df[numeric_cols].copy()
        for c in numeric_cols:
            num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
            med = num_df[c].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            num_df[c] = num_df[c].fillna(med)
        # apply spatial weight if requested
        if spatial_weight != 1.0:
            for s in ("latitude_num", "longitude_num"):
                if s in num_df.columns:
                    try:
                        num_df[s] = num_df[s].astype(float) * float(spatial_weight)
                    except Exception:
                        pass
        mat_parts.append(num_df.values.astype(float))

    # categorical handling
    if cat_cols:
        cat_df = df[cat_cols].astype(str).fillna("~NA~")
        enc = OrdinalEncoder(dtype=float)
        try:
            cat_enc = enc.fit_transform(cat_df)
        except Exception:
            # fallback: label codes per column
            cat_enc = []
            for c in cat_cols:
                codes = pd.Categorical(cat_df[c]).codes
                cat_enc.append(codes)
            cat_enc = np.vstack(cat_enc).T
        mat_parts.append(cat_enc.astype(float))

    if not mat_parts:
        raise RuntimeError("No valid features found to construct feature matrix. Specify --features explicitly.")

    X = np.hstack(mat_parts)
    return X, final_feats, types


def scale_matrix(X: np.ndarray, scale: bool = True) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    if not scale:
        return X, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def run_clustering(X: np.ndarray, method: str, random_state: int = 0, **kwargs):
    method = method.lower()
    if method == "kmeans":
        n_clusters = int(kwargs.get("n_clusters", 10))
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        meta = {"model": "kmeans", "n_clusters": n_clusters}
        return labels, meta
    if method == "gmm":
        n_components = int(kwargs.get("n_clusters", 10))
        model = GaussianMixture(n_components=n_components, random_state=random_state)
        labels = model.fit_predict(X)
        meta = {"model": "gmm", "n_components": n_components}
        return labels, meta
    if method == "dbscan":
        eps = float(kwargs.get("eps", 0.5))
        min_samples = int(kwargs.get("min_samples", 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        meta = {"model": "dbscan", "eps": eps, "min_samples": min_samples}
        return labels, meta
    raise RuntimeError(f"Unknown clustering method: {method}")


def summarize_patterns(df: pd.DataFrame, top_n: int = 20):
    print("Pattern summary (top clusters by size):")
    vc = df["pattern_label"].value_counts().head(top_n)
    print(vc.to_string())
    print(f"Total rows: {len(df)}, unique patterns: {df['pattern_label'].nunique()}")


def label_patterns(df: pd.DataFrame,
                   features: List[str],
                   method: str = "kmeans",
                   n_clusters: int = 10,
                   eps: float = 0.5,
                   min_samples: int = 5,
                   scale: bool = True,
                   random_state: int = 0,
                   spatial_weight: float = 1.0) -> pd.DataFrame:
    """
    Perform clustering and attach pattern columns to a copy of df. Returns the labeled DataFrame.
    """
    X, used_feats, feat_types = prepare_matrix(df, features, spatial_weight=spatial_weight)
    Xs, scaler = scale_matrix(X, scale=scale)

    # cluster
    labels, meta = run_clustering(Xs, method=method, random_state=random_state,
                                  n_clusters=n_clusters, eps=eps, min_samples=min_samples)

    out = df.copy().reset_index(drop=True)
    out["pattern_id"] = labels.astype(int)

    # pattern_label (string) and pattern_size
    def label_name(pid: int) -> str:
        if pid == -1:
            return "noise"
        return f"pattern_{int(pid)}"

    out["pattern_label"] = [label_name(int(p)) for p in out["pattern_id"].to_numpy()]

    sizes = out["pattern_id"].value_counts().to_dict()
    out["pattern_size"] = out["pattern_id"].map(lambda p: int(sizes.get(int(p), 0)))

    out["pattern_method"] = f"{meta.get('model')}" + ("_" + str(meta) if meta else "")
    out["pattern_features"] = ",".join(used_feats)

    # Compute simple pattern_name summarizing top discriminative features per cluster
    # For numeric features: compute ANOVA F (if possible) and cluster medians; for categorical use mode.
    numeric_feats = [f for f in used_feats if feat_types.get(f) == "numeric"]
    categorical_feats = [f for f in used_feats if feat_types.get(f) == "categorical"]

    # compute medians per cluster for numeric feats
    if numeric_feats:
        median_df = out.groupby("pattern_id")[numeric_feats].median()
        global_median = out[numeric_feats].median()
        global_std = out[numeric_feats].std().replace(0, np.nan).fillna(1.0)
        # z-scores of medians
        z = (median_df - global_median) / global_std
        abs_z = z.abs()
    else:
        median_df = pd.DataFrame()
        abs_z = pd.DataFrame()

    # categorical modes
    cat_modes = {}
    for c in categorical_feats:
        cat_modes[c] = out.groupby("pattern_id")[c].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan)

    # combined ranking (use mutual info if possible)
    feature_scores = pd.Series(0.0, index=used_feats)
    try:
        if len(np.unique(labels)) > 1:
            mi = mutual_info_classif(np.nan_to_num(Xs), labels, random_state=random_state)
            mi_series = pd.Series(mi, index=used_feats).sort_values(ascending=False)
            for i, f in enumerate(mi_series.index):
                feature_scores[f] += (len(mi_series) - i)
    except Exception:
        pass

    # additionally weight numeric ANOVA ranks
    try:
        if numeric_feats and len(np.unique(labels)) > 1:
            fvals, pvals = f_classif(out[numeric_feats].fillna(0).values, labels)
            anova_series = pd.Series(fvals, index=numeric_feats).sort_values(ascending=False)
            for i, f in enumerate(anova_series.index):
                feature_scores[f] += (len(anova_series) - i) * 1.5
    except Exception:
        pass

    # Build pattern_name from a combination of top numeric discriminators (by abs z) and categorical modes
    pattern_rows = []
    for pid in sorted(out["pattern_id"].unique()):
        size = int(sizes.get(int(pid), 0))
        if pid == -1:
            pname = "noise"
        else:
            parts = []
            # numeric top features by abs z-score for this cluster
            if not abs_z.empty and int(pid) in abs_z.index:
                z_row = abs_z.loc[int(pid)].sort_values(ascending=False)
                for feat in z_row.index[:2]:
                    try:
                        val = median_df.at[int(pid), feat]
                        parts.append(f"{feat}={val:.2f}")
                    except Exception:
                        pass
            # categorical modes (up to 2)
            cat_parts = []
            for c in list(cat_modes.keys())[:2]:
                try:
                    modev = cat_modes[c].at[int(pid)]
                    if pd.notna(modev):
                        cat_parts.append(f"{c.split('_')[0]}={modev}")
                except Exception:
                    pass
            # fallback to top feature_scores if still empty
            if not parts:
                for f in feature_scores.dropna().index[:2]:
                    if f in numeric_feats:
                        try:
                            val = median_df.at[int(pid), f]
                            parts.append(f"{f}={val:.2f}")
                        except Exception:
                            pass
            name_parts = parts + cat_parts
            if not name_parts:
                pname = f"pattern_{int(pid)}"
            else:
                pname = f"pattern_{int(pid)} | " + "; ".join(name_parts)
        pattern_rows.append({"pattern_id": int(pid), "pattern_size": size, "pattern_name": pname})

    pattern_attributes = pd.DataFrame(pattern_rows).set_index("pattern_id").sort_values("pattern_size", ascending=False)
    # attach top overall features (for context)
    pattern_attributes["top_features"] = ",".join(feature_scores.sort_values(ascending=False).index[:10].tolist())

    # map back to rows
    id_to_name = pattern_attributes["pattern_name"].to_dict()
    out["pattern_name"] = out["pattern_id"].map(lambda p: id_to_name.get(int(p), ("noise" if int(p) == -1 else f"pattern_{int(p)}")))

    # ensure columns ordering convenience
    cols_to_ensure = ["pattern_id", "pattern_label", "pattern_name", "pattern_size", "pattern_method", "pattern_features"]
    for c in cols_to_ensure:
        if c not in out.columns:
            out[c] = np.nan

    # attach pattern_attributes for convenience as returned attribute (not a column)
    out.attrs["pattern_attributes"] = pattern_attributes.reset_index()

    return out


def parse_feature_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Cluster events_celestial.csv rows into pattern groups and write labeled CSV.")
    p.add_argument("--input", "-i", required=True, help="Input events_celestial.csv")
    p.add_argument("--out", "-o", required=True, help="Output CSV with pattern columns added")
    p.add_argument("--features", "-f", help="Comma-separated list of feature columns to use (default: auto-detect)")
    p.add_argument("--method", "-m", default="kmeans", choices=["kmeans", "dbscan", "gmm"], help="Clustering method")
    p.add_argument("--n-clusters", type=int, default=10, help="Number of clusters for kmeans/gmm")
    p.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps (if using dbscan)")
    p.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples (if using dbscan)")
    p.add_argument("--no-scale", dest="scale", action="store_false", help="Disable StandardScaler (don't scale features)")
    p.add_argument("--random-state", type=int, default=0, help="Random state for reproducibility")
    p.add_argument("--spatial-weight", type=float, default=1.0, help="Multiply latitude_num/longitude_num by this factor to affect spatial influence")
    args = p.parse_args()

    print("Reading input:", args.input)
    df = pd.read_csv(args.input, low_memory=False)

    # Ensure latitude_num/longitude_num exist
    if "latitude_num" not in df.columns and "latitude" in df.columns:
        df["latitude_num"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "longitude_num" not in df.columns and "longitude" in df.columns:
        df["longitude_num"] = pd.to_numeric(df["longitude"], errors="coerce")

    req_feats = parse_feature_list(args.features)
    features = detect_feature_columns(df, requested=req_feats)
    if not features:
        print("No features found for clustering. Exiting.")
        sys.exit(1)

    print("Using features:", features)
    labeled = label_patterns(
        df,
        features=features,
        method=args.method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
        scale=args.scale,
        random_state=args.random_state,
        spatial_weight=args.spatial_weight,
    )

    # Save output CSV
    labeled.to_csv(args.out, index=False)
    print(f"Wrote output to {args.out} (rows: {len(labeled)})")

    # Save pattern attributes CSV if available
    patt_attrs = labeled.attrs.get("pattern_attributes", None)
    if patt_attrs is not None:
        patt_out = args.out.rsplit(".", 1)[0] + ".pattern_attributes.csv"
        patt_attrs.to_csv(patt_out, index=False)
        print(f"Wrote pattern attributes to {patt_out}")

    summarize_patterns(labeled)


if __name__ == "__main__":
    main()