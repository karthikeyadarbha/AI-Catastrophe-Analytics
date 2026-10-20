# TG-SAI audit: what was found and what changed

**Short version.** When the model is rebuilt with matched negative examples and correct sky positions, it
cannot tell great-earthquake times from random times (AUC about 0.50; 0.54 in cross-validation, 0.42-0.44 on a
time split). The risk labels in the main README (below 50 %, 75-85 %, 85 % and above) therefore come from the
way the original features and training examples were built, not from skill. Nothing here says earthquakes
cannot be forecast; it says this feature set does not do it.

## Problems found in the original pipeline
| # | Problem | Evidence |
|---|---|---|
| 1 | Syzygy, Saturn, Jupiter and Mars features are 7-day peaks that sit near their maximum for about a week around every new/full moon | instantaneous vs peak correlation 0.15; features near 1 for most days |
| 2 | Planet "alignments" use crude mean-longitude formulas | `Sid_Topo_Mars/Jup/Sat_Lon` in the training file differ from true positions by a median of 25, 7 and 5 degrees (up to 52) |
| 3 | `Moon_Dist_km` is constant (384,467 km) in all 104 rows | perigee cannot be computed from it |
| 4 | `Lunar_Torque_Index` = |Moon declination| / 10, not a separate quantity | max difference 0.00005 |
| 5 | `Topo_Node_Lon` is tropical while all other longitudes are sidereal | matches the tropical mean node to 0.00 degrees; refitting with a sidereal node changed AUC by about 0.01 |
| 6 | Training file holds positives only; negatives were sampled differently from the earthquakes | no negative rows in the file |

## What was done
* `analysis/tgsai_corrected/` rebuilds the features (true planets, sidereal node, computed Moon distance,
  instantaneous values), regenerates the training file, and evaluates with matched negatives, 5-fold CV, a time
  split and shuffled labels.
* The model was also scored once, without retraining, on older earthquakes (1900-1989) and on the repository's
  own 1850-1950 table: AUC 0.50-0.51.
* The Deadly 15 earthquakes were re-scored with the corrected model: AUC 0.55 (range 0.51-0.60), not
  distinguishable from chance with 15 events.
* Vedic rules (Parasara Tantra, Brihat Samhita) were tested on 15, 1,270, 9,368, 6,111 and 670 earthquakes with
  rules frozen in advance: none passes. See the two PDFs and `preregistrations/`.

## Reproduce
```bash
cd analysis/tgsai_corrected
pip install -r requirements.txt
python build_training_file.py ../../notebooks/megaquake/Inputs_Mega_Quake_Topocentric_Analysis.csv Inputs_Mega_Quake_Topocentric_Analysis_corrected.csv
python train_and_evaluate.py Inputs_Mega_Quake_Topocentric_Analysis_corrected.csv --score deadly15_events.csv --reps 10
```

## Caveats
* `ephemeris.py` is a self-written Meeus/Standish ephemeris, valid about 1800-2050, checked against an
  independent tool (planets within 0.17 degrees, Moon and Sun within 0.01 degrees). Swap in Swiss Ephemeris if you
  need higher precision.
* Only 102 great earthquakes (M8+, 1850-2025) are in the training file, so only large effects could be seen.
* Earthquake catalogues for the wider tests came from USGS (including reviewed ISC-GEM records back to 1900).
