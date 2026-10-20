# Experiment 1: Rule-First Study
Starts from rules written in the texts and asks whether real earthquakes follow them. Run from a working directory as described in `../README.md`.

| Folder | Report section | What it does | Main outputs (in `results/`) |
|---|---|---|---|
| `01_deadly15/` | 4, 6.6 | Deadly 15: Parasara Tantra Ch.3-16 features (`ch14_16.py`, `ch3_10.py`), eclipses (`ecl_test.py`), Cunningham regions (`reg.py`), circles under all treatises (`d15_all.py`) | `ch14_16_tests.csv`, `eclipse_proximity_results.csv`, `deadly15_circles_all_treatises.csv` |
| `02_rules_large_sets/` | 5, 12 | 23 pre-registered rules: M6.5+ set (`big_test.py`), M5.5+ set (`band55.py`, `shift55.py` time-shift null), older sets (`run_hist.py A` or `B`) | `prereg_large_sample_results.csv`, `m55_23rules_results.csv`, `hist_results_A/B.csv` |
| `03_brihat_samhita/` | 6.2-6.4, 6.6, 7 | Brihat Samhita Ch.32 circles, time of day, felt distance, eclipse window (`bs_en.py`; `bs15.py` variants), Vayu and crescent tests (`extra*.py`), true-horizon crescent (`run_cres.py`), Ch.17 Mercury-Jupiter (`bs_merjup.py`) | `bs_english_crosscheck_results.csv`, `crescent_results.csv`, `brihat_samhita_ch17_*.csv` |
| `04_adbhuta_sagara/` | 6.5 | 13 tests: authority lists, disputed nakshatras, yama clock (night reversed or same order), twilight, felt distance; `tw_check.py` shows the twilight artifact | `adbhuta_sagara_results.csv` |
| `05_repeat_shocks_verse18_7/` | 10 | Repeat-shock lags (PT 18.7, BS 32.34, AS) against aftershock decay | `verse18_7_*.csv` |
| `06_regional/` | 8 | Six regional belts x seven rules | `regional_results.csv` |
| `07_ayanamsa/` | 11 | Named ayanamsa systems and a 29-offset sweep | `ayanamsa_*_tests.csv` |
| `08_tgsai/` | 9 | TG-SAI retraining with matched negatives (`tgsai_fix.py`), older sets (`tgsai_hist.py`), supplied training file audit and refit (`an.py`, `refit.py`, `score15.py`; copy the repo's `notebooks/megaquake/Inputs_Mega_Quake_Topocentric_Analysis.csv` to `train.csv`) | `tgsai_retrain_results.csv`, `tgsai_historical_results.csv`, `refit.csv` |

Notes: `ch14_16.py` uses sidereal longitudes for the Deadly 15 that were taken from the astral-ephemeris tool; they are stored in `shared/inputs/earthquakes_parashara_tantra_mandalas.csv`. `ch3_10.py`, `bs15.py`, `tw_check.py` and `score15.py` reuse code from a sibling script, so keep the folder together. Pre-registrations: `docs/audit/preregistrations/`.
