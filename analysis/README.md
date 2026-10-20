# Vedic earthquake experiments: analysis code

Code behind the two reports in `docs/audit/` (PDFs): the **Rule-First Study** (experiment 1) and the **Data-First Study** (experiment 2). PDF-building code is not included. The corrected TG-SAI pipeline is in `analysis/tgsai_corrected/`.

## Layout
| Folder | Contents |
|---|---|
| `shared/` | Own ephemeris (`myeph.py`: Meeus Sun/Moon, Standish planets, mean node, Lahiri ayanamsa), eclipse finder, background-grid helper (`hist.py`), crescent helpers, USGS fetch scripts, feature builder (`big55.py`) |
| `shared/inputs/` | Small inputs so scripts run without a download: Deadly 15 table, M6.5+ catalogue (`big.csv`), Modern M5.5+ input, Older set 1 and 2 (declustered), 1964-1989 catalogue |
| `experiment1_rule_first/` | Rules taken from the texts (Parasara Tantra, Brihat Samhita, Adbhuta Sagara, Cunningham) and tested on the Deadly 15 and larger sets |
| `experiment2_data_first/` | Scans that start from the data and look for patterns, then replication and gap tests |

## Setup
`pip install numpy pandas scipy scikit-learn requests` (see `requirements.txt`).
Scripts read inputs from, and write results to, the **current directory** and a `results/` folder. Create a scratch folder (git-ignored) and run from there:

```
cd analysis && mkdir -p workdir/results && cd workdir
cp ../shared/inputs/*.csv . ; cp ../shared/inputs/earthquakes_parashara_tantra_mandalas.csv results/
# period-matched background grid, then features for events and grid (a few seconds):
python ../shared/make_background.py
python ../shared/big55.py
# example:
python ../experiment1_rule_first/04_adbhuta_sagara/adsagara.py     # -> results/adbhuta_sagara_results.csv
```
Scripts find `shared/` by themselves, so any working directory works as long as the data files are in it. To re-download catalogues instead of using `shared/inputs/`, run `shared/fetch55b.py`, `shared/fetch_hist.py`, `shared/fetch_old.py`, `shared/fetch_m7.py` (USGS FDSN service; needs network), then `shared/decl55.py`.

## What is not included
The Modern M6.5+ scripts (`big_test.py`, `extra*.py`, `ayan.py`, `stage1_*.py`) also need sidereal and tropical features that were produced with the astral-ephemeris tool: `bg_sid.csv` (with features), `bg_trop.csv`, `quakes_m65_sid.csv`, `quakes_m65_trop.csv` (about 7.5 MB each). They are not committed; regenerate them with the tool (`enrich_csv_dataset`, Lahiri sidereal and tropical) or ask for the files. Everything on the Modern M5.5+ set, the older sets and the Adbhuta Sagara and Brihat Samhita tests runs from `shared/inputs/` alone; the Adbhuta Sagara script was re-run from a clean folder and reproduces `adbhuta_sagara_results.csv` exactly.

## Conventions
Sidereal frame = Lahiri; nakshatra index Ashwini=0 ... Revati=26. Background = 6-hourly grid over the same years, each time shifted by a random 0-6 h, with event locations shuffled onto background times. Seeds are fixed in the scripts. Rules were written down before testing (see `docs/audit/preregistrations/`); `preregistration_*` files are the contract, scripts are the implementation.
