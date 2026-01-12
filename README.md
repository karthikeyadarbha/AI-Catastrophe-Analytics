```markdown
# Celestial Patterns — README

This repository generates celestial features for earthquake events, creates synthetic control anchors, clusters the combined anchors into human-meaningful "patterns", and provides visualizations and exports to support downstream analysis.

This README explains:
- the repository layout,
- prerequisites,
- step-by-step instructions to create and activate a virtual environment,
- how to run the pipeline (generate dataset → label patterns → visualize),
- how to reproduce the interactive notebook outputs and expanded CSVs,
- quick troubleshooting notes.

---

## Repository layout (important files)

- `src/generate_celestial_dataset.py`  
  Generates anchors (events + synthetic controls) and computes celestial/sidereal/JPL features. Produces `events_celestial.csv` by default.

- `src/label_patterns.py`  
  Clusters rows from `events_celestial.csv` into patterns and writes a patterned CSV. Includes spatial weighting and keeps `latitude_num`/`longitude_num`.

- `run_jpl.py`  
  CLI helper for computing JPL features for a single epoch or augmenting a CSV with JPL columns.

- `notebooks/visualize_patterns.ipynb`  
  Comprehensive notebook that re-clusters (optionally), derives human-readable `pattern_name` values, visualizes results (map, projection, histograms, heatmaps), and exports per-row and per-pattern expanded CSVs.

- `requirements.txt`  
  Project dependencies (un-pinned). See below for install instructions.

---

## Quick prerequisites

- Python 3.8+ (3.10 recommended)
- Git (optional)
- Internet access for:
  - Skyfield to download DE files on first use (e.g. `de421.bsp`),
  - astroquery JPL Horizons (optional; network + rate limits).

If you plan to use `pyswisseph` (swisseph), installation usually works via pip wheels for common platforms; otherwise consult pyswisseph docs.

---

## Create and activate a virtual environment (recommended)

Linux / macOS (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell)
```powershell
python -m venv .venv
# If activation is blocked, allow for current process:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Conda (optional)
```bash
conda create -n celest python=3.10 -y
conda activate celest
pip install -r requirements.txt
```

Notes:
- If you won't use JPL Horizons API, you may omit `astroquery`. The code falls back to Skyfield.
- If any package fails to build (e.g., pyswisseph), look for platform wheel or follow package-specific installation instructions.

---

## Step-by-step execution

All commands assume you have activated the virtual environment.

1) Prepare input CSV
- Ensure your earthquake input CSV contains at least:
  - `time` (ISO-8601 or parseable),
  - `latitude` and `longitude`.
- If present, the generator will preserve `depth`, `mag`, `magType`, `depthError`, `magError`.

2) Generate the celestial dataset (events + synthetic controls)

Example (quick smoke test — only first N anchors):
```bash
# run generator with defaults; adjust --input and --out as needed
python src/generate_celestial_dataset.py \
  --input 1850-1950-EQData-MAG5.csv \
  --out events_celestial.csv \
  --controls-per-event 2 \
  --max-rows 50
```

Important notes:
- Output columns will include original event fields (time, latitude, longitude, depth, mag, magType, depthError, magError if present), plus:
  - `label` (1 = original event, 0 = synthetic control),
  - `is_synthetic` (True for synthetic controls),
  - `source_event_index`,
  - `_parsed_time_` and `_jd_`,
  - JPL/Skyfield features (e.g., `Sun_ra_hours`, `Moon_dec_deg`, `Sun_distance_km`, ...),
  - sidereal/swisseph features if swisseph is installed (e.g., `Sun_sid_long`),
  - altitude, is_combust flags, etc.

3) Label patterns (clustering)

- Use `src/label_patterns.py` to cluster and attach pattern columns.
- Example: run with k-means (12 clusters), include spatial influence doubled:
```bash
python src/label_patterns.py \
  --input events_celestial.csv \
  --out events_celestial.patterned.csv \
  --method kmeans \
  --n-clusters 12 \
  --spatial-weight 2.0
```

Behavior:
- Automatically detects features (includes `latitude_num` and `longitude_num` by default).
- Outputs added columns:
  - `pattern_id` (int),
  - `pattern_label` (string),
  - `pattern_name` (human-friendly description),
  - `pattern_size`, `pattern_method`, `pattern_features`.
- Also creates a `<out>.pattern_attributes.csv` (one row per pattern) in the same step.

4) Run the visualization notebook

- Start Jupyter Notebook (or JupyterLab) from the repository root:
```bash
jupyter notebook notebooks/visualize_patterns.ipynb
# or
jupyter lab notebooks/visualize_patterns.ipynb
```

- The notebook:
  - can re-run clustering (comprehensive workflow) or use existing `events_celestial.patterned.csv`,
  - creates interactive Folium map HTML (`events_celestial_patterns_comprehensive_map.html`),
  - writes `events_celestial.comprehensive.patterned.csv` and `events_celestial.pattern_attributes.comprehensive.csv`,
  - writes expanded CSV outputs:
    - `events_celestial_pattern_expanded_row_level.csv` — a per-row CSV that includes the top-N global features with actual row-level values as additional columns,
    - `events_celestial_pattern_expanded_pattern_representatives.csv` — one representative row per pattern with actual values for top features,
    - `events_celestial_pattern_legend.json`.

If you want the notebook to use the already produced skirted pattern CSV rather than recompute clustering, open it and set the `INPUT_CSV` and control variables at the top accordingly.

5) Inspect expanded CSVs
- `events_celestial_pattern_expanded_row_level.csv` contains, for each row, columns:
  - `top_feature_1_name`, `top_feature_1_value`, ..., `top_feature_N_name`, `top_feature_N_value`,
  - plus `pattern_size` and `pattern_top_features` for context.
- `events_celestial_pattern_expanded_pattern_representatives.csv` contains one row per pattern with representative values for the selected top features.

---

## Recommended workflows and tips

- If you will compute many JPL/Horizons epochs:
  - Prefer the Skyfield + local DE file for performance.
  - If using astroquery/Horizons, consider caching queries or precomputing JPL features for unique epochs to avoid rate limits.

- Spatial weighting:
  - Use `--spatial-weight` to upweight or downweight geographic coordinates versus celestial features.
  - Because geographic coordinates are on different scales, try scaling and experimenting (e.g., 0.5, 2.0) and inspect cluster interpretability.

- Feature engineering ideas:
  - Convert angular features to sine/cosine pairs (to handle wrap-around for longitudes/RA).
  - Build relative/derived features such as angular separation between Moon and Sun, or scaled heliocentric distances.

- Reproducibility:
  - After validating the pipeline, pin dependency versions:
    ```bash
    python -m pip freeze > requirements-pinned.txt
    ```

---

## Troubleshooting

- Import errors:
  - Ensure you are using the same Python interpreter (inside the venv). Check `which python` (or `where python` on Windows) and `pip list`.
- pyswisseph install issues:
  - Look for prebuilt wheels; if not available, consult the package docs for C build prerequisites.
- Skyfield DE download fails:
  - Manually download `de421.bsp` (or your desired DE) and place it in the repo root or specify `--de-file` to point to its path.
- Notebook memory or UI slow:
  - Sample the dataset when viewing the map (notebook limits drawing to `MAX_MAP_POINTS`).

---

## Files produced by the pipeline (summary)

- events_celestial.csv — events + synthetic anchors and computed celestial features.
- events_celestial.patterned.csv — per-row patterns (if labeled via `label_patterns.py`).
- events_celestial.comprehensive.patterned.csv — comprehensive/reclustered patterned CSV from the notebook.
- events_celestial.pattern_attributes.*.csv — per-pattern summaries (one row per pattern).
- events_celestial_patterns_comprehensive_map.html — interactive map for exploration.
- events_celestial_pattern_expanded_row_level.csv — per-row expanded top-feature columns (actual row-level values).
- events_celestial_pattern_expanded_pattern_representatives.csv — per-pattern representative rows with top-feature values.

---

## If you want help with any of these:
- Producing a pinned requirements file for a specific Python version,
- Adding an on-disk cache for JPL Horizons queries,
- Refactoring compute_jpl_features into a shared module,
- Adding a small Streamlit dashboard that allows pattern selection and exploration,

tell me which item and I will prepare the code / CI workflow.

Thank you!