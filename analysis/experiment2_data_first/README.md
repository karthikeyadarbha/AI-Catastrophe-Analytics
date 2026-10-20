# Experiment 2: Data-First Study
Starts from the earthquake catalogues, scans every sky variable for over-represented bins, and then replicates what looks interesting. Run from a working directory as described in `../README.md`.

| Script | Report section | What it does | Main outputs (in `results/`) |
|---|---|---|---|
| `stage1_bottom_up_scan.py` | 2 | Scan of nakshatra, sign, tithi, planet separation and flag bins on the Modern M6.5+ set, against a matched background | `bottom_up_bin_scan.csv` |
| `stage1_slow_planet_recheck.py` | 2 | Re-check of the lead bins (slow planets) with a matched baseline | `prereg_slow_planet_recheck_matched_baseline.csv` |
| `stage1_sun_nakshatra_1964_1989.py` | 2, 6 | Frozen follow-up of the Sun-in-Vishakha lead on 1964-1989 | `sun_nakshatra_1964_1989_followup.csv` |
| `stage3_gap_tests.py`, `stage3_gap_time_shift.py` | 5 | Gap tests and time-shift null on the Modern M5.5+ set | `gap_tests_primary.csv`, `gap_slow_planet_time_shift_check.csv` |
| `stage4_scan_older_eras.py` | 6 | Same scan on Older set 1 and Older set 2 | `hist_bin_scan_comparison.csv`, `hist_bin_scan_summary.csv` |

The Modern M5.5+ replication of the 23 rules (Stage 2) is `../experiment1_rule_first/02_rules_large_sets/band55.py`, and the older-era rule replication is `run_hist.py`; both are shared with experiment 1. Pre-registrations: `docs/audit/preregistrations/`.
