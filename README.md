# 🌍 Topocentric Gravitational-Seismic AI (TG-SAI)

> **Project Title:** AI-Driven Tectonic Stress Forecasting  
> **Architecture:** Vectorized 7-Parameter Topocentric ML Engine  
> **Forecast Horizon:** 2026 – 2031  

---

> ## ⚠️ Audit notice (October 2026)
> An independent audit found that the risk labels below are **not supported**: once the features use true
> planetary positions and the training examples are matched, the model scores at chance (AUC about 0.50).
> See [`docs/audit/README_AUDIT.md`](docs/audit/README_AUDIT.md) for the findings, the corrected pipeline in
> [`analysis/tgsai_corrected/`](analysis/tgsai_corrected/) and the two full reports.


## 📖 Executive Summary
The **TG-SAI** is a high-performance predictive system designed to identify windows of extreme seismic risk. Unlike traditional global models, this engine uses **Topocentric Physics** to calculate the exact gravitational lift and torque exerted on specific tectonic coordinates by the Moon, Sun, and planets.

By combining **Orbital Mechanics** with **Random Forest Machine Learning**, the system distinguishes between routine celestial alignments and rare "Black Swan" gravitational configurations that historically precede mega-thrust events ($8.0+ \ M_w$).

---

## 🛠️ Technical Methodology

### **1. The Physics Engine (The "Trigger")**
The engine operates on the principle that celestial bodies provide the final incremental force required to rupture a pre-strained fault (Triggering vs. Loading).
* **Vectorization:** Math operations are performed using `numpy` C-speed matrix broadcasting, calculating millions of daily vectors across the global tectonic grid in seconds.
* **Tidal Force Ratio:** The AI independently validated the Newtonian tidal ratio (~2.2:1), weighing Lunar Zenith Pull significantly higher than Solar Pull.

### **2. The 7-Feature Vector Space**
The AI evaluates every tectonic coordinate using these primary physical vectors:
1.  **Lunar Syzygy:** Sun-Moon alignment (Spring Tides).
2.  **Saturn Alignment:** Gas giant gravitational resonance.
3.  **Jupiter Alignment:** Primary planetary mass influence.
4.  **Mars Alignment:** Close-proximity geometrical resonance.
5.  **Lunar Perigee:** Inverse-square multiplier ($1/r^3$) for tidal force based on distance.
6.  **Local Lunar Zenith Lift:** Direct vertical pull relative to the fault's latitude.
7.  **Local Solar Zenith Lift:** Direct vertical solar pull relative to the fault's latitude.



---

## 🚀 Implementation Workflow

### **Step 1: Tectonic Grid Generation**
Generates a high-speed coordinate matrix of the Earth's primary fault lines.
* **Resolution:** $0.5^\circ$ (~55km intervals).
* **Coverage:** Subduction Zones (Ring of Fire) and Global Transform Faults (Turkey, San Andreas, Himalayas).

### **Step 2: Training & Hard-Negative Mining**
* **Grounding:** Trained on the `Inputs_Mega_Quake_Topocentric_Analysis.csv` historical dataset.
* **Noise Reduction:** The AI is fed 10,000 "Hard Negatives"—dates with high planetary alignment but zero seismic activity—to ensure it only fires for "Perfect Storms."

### **Step 3: Vectorized Spatial Scan**
* **Logic:** The engine calculates the gravitational state for every point on the grid for every day in the forecast window (2026–2031).
* **Speed:** Uses `n_jobs=-1` for multi-core parallel processing.

### **Step 4: Peak Isolation & Epicenter Identification**
* **Spatial Isolation:** Uses `.idxmax()` to identify the **Gravitational Epicenter** (the exact coordinate of peak strain).
* **Temporal Isolation:** Uses `scipy.signal.find_peaks` to filter for a 21-day "Clear Window," preventing duplicate alerts for the same lunar cycle.

---

## 📖 Key Terminology & Interpretation

| Term | Definition |
| :--- | :--- |
| **Topocentric Zenith** | The point directly overhead ($90^\circ$). This maximizes vertical lift, reducing tectonic friction (Normal Stress reduction). |
| **Julian Date (JD)** | The continuous day count from the J2000.0 epoch; used for precise orbital tracking. |
| **Gravitational Epicenter** | The specific coordinate where planetary torque is at its mathematical maximum for a given fault. |

### **Risk Probability Legend**
* **⚪ < 50% [NORMAL]:** Background stress. No significant celestial help. Quakes here (like 2001 Bhuj) are driven primarily by internal strain.
* **🟡 75% - 85% [HIGH]:** Significant unweighting of the fault. The "Trigger" is active. (e.g., 2004 Sumatra).
* **🔴 85%+ [EXTREME]:** **"The Black Swan."** Global alignments match the configuration of historical $9.0+$ events (e.g., 2011 Tohoku).



---

## 📂 Output & Visualization
The system generates map-ready CSV files containing:
* `Risk_Date`: Day of peak gravitational stress.
* `Threatened_Zone`: Tectonic segment name.
* `Latitude` & `Longitude`: Exact "Bullseye" coordinates.
* `Peak_Stress_Probability`: The AI confidence score.

**Visual Tool:** Use the provided `folium` script to convert these coordinates into an interactive global heat map.


# 🚨 ZERO HOUR REPORT: MEGA-QUAKE FORECAST (2026–2031)
**Analytical Engine:** Vectorized 7-Parameter Topocentric AI  
**Data Integrity:** Unified DuckDB Ensemble (`mega_quake_unified.db`)  
**Report Date:** 2026-02-16  

---

## 📈 Global Strategic Overview
This report identifies the highest-risk gravitational anomalies for the next five years. The system isolates windows where **Planetary Resonance** and **Lunar Zenith Lift** create a "Trigger Environment" for major tectonic ruptures ($M_w 7.5+$).

---

## 📅 High-Risk Chronology (Top 5 Events)

| Risk Date | Threatened Zone | Lat, Lon | AI Confidence | Stress Profile |
| :--- | :--- | :--- | :--- | :--- |
| **2028-05-12** | **Japan Trench (Tohoku)** | `38.50, 142.50` | **91.4% [🔴]** | Triple Alignment + Zenith Bullseye |
| **2029-11-20** | **Sumatra-Java Trench** | `-3.50, 101.50` | **88.7% [🔴]** | Perigee Supermoon + Syzygy |
| **2027-02-04** | **East Anatolian (Turkey)** | `37.50, 37.00` | **86.2% [🔴]** | Mars Resonance + Solar Zenith |
| **2030-08-15** | **Himalayan Front (Nepal)** | `28.00, 85.00` | **84.9% [🟡]** | Saturn-Jupiter Vector Overlap |
| **2026-12-24** | **San Andreas (California)**| `34.50, -118.00`| **82.1% [🟡]** | Lunar Declination Peak |

---

## 🔬 Physics Breakdown: The Trigger Mechanisms

### **1. The Japan 2028 "Black Swan"**
* **Mechanism:** The Moon reaches a specific declination that aligns exactly with the 38.5°N latitude of the Tohoku segment. 
* **Tidal Effect:** Combined with a Sun-Mars alignment, the vertical pull (Zenith Lift) reaches a mathematical maximum.
* **Tectonic Impact:** Reduction of **Normal Stress** on the subduction interface, allowing the Pacific Plate to slip.
[Image of the Japan Trench subduction zone showing tectonic plate interaction]

### **2. The Sumatra 2029 Perigee Surge**
* **Mechanism:** The Moon reaches Perigee (closest approach) within 12 hours of the Full Moon.
* **Tidal
## Astral MCP server (platform)
The ephemeris platform behind the Vedic-astrology experiments lives in [`astral-mcp-server/`](astral-mcp-server/) (MIT licence). The experiment code is in [`analysis/`](analysis/) and the reports in [`docs/audit/`](docs/audit/).

The Indian-method platform (Āryabhaṭīya, Tantrasaṅgraha, Gaṇitayuktibhāṣā: a verse-referenced rule registry plus a computation engine) lives in [`indic-ganita-engine/`](indic-ganita-engine/) (MIT licence, frozen v1.1.0, see its `FREEZE.md`).

---

# Data pipeline and study conclusions (Bronze → Silver → Gold)

Source books go in at the bottom (Bronze), become checked, searchable rules (Silver), and are tested against
earthquakes (Gold). Every step is recorded in a tamper-evident audit log, and every later layer refuses to
build if an earlier one fails its checks.

| Layer | Folder | What it holds |
|---|---|---|
| Bronze | `bronze_text_sources/` | Scans registered by fingerprint (SHA-256), extracted page text, hash-chained audit log |
| Silver | `silver_text_rules/` | `silver.duckdb`: texts, pages, the engine's 85 rules, and the testable claims (Experiment 1: 23, Experiment 2 stage 5: 54) with the results the studies published |
| Gold | `gold_experiment1/`, `gold_experiment2/` | Re-runs of the studies from Silver's rules, checked against the committed results |

## 1. Conclusions of the two experiments

Full reports: `docs/audit/RuleFirst_Study_Vedic_Earthquake_Patterns.pdf` and `docs/audit/DataFirst_Study_Vedic_Earthquake_Patterns.pdf`.

**Experiment 1, Rule-First (start from the texts' rules, test them on earthquakes).** No rule, sky measure or model tested
predicted earthquakes better than chance. 23 rules were written down before testing; on the 638 hold-out earthquakes
(M6.5+, 2008 onward) none passed the strict bar of p < 0.0022 (best p = 0.039). The larger M5.5+ set (9,368 events) did not rescue
any rule (closest p = 0.0030). The Adbhuta Sāgara, the rest of the Brihat Saṃhitā, crescent-Moon omens, regional belts, verse 18.7,
the choice of ayanāṃśa and older earthquakes (1900–1989) were all null, and the TG-SAI model scores about 0.50 AUC (coin-toss level).

**Experiment 2, Data-First (start from the earthquakes, look for what stands out, then compare with the texts).** A scan of 433 sky
bins found only a faint agreement between the two halves of the data (r = 0.105). Its top lead, the Sun in Viśākhā, did not replicate
on 1964–1989 (1.15×, p = 0.24) or on 8,098 new earthquakes. Combinations and slow-planet factors all failed their bars. The 54
bins defined by the texts themselves gave 6 nominal hits in 1990–2007, all one effect (Moon in the Indra circle, about 7% low), and none held in 2008–2026.

**Overall:** at these sample sizes and with these pre-registered tests, nothing in the source texts shows a reliable link to
earthquakes. This is evidence of absence of a detectable link, not a proof that no weaker or differently defined effect exists.

**Reproducibility check (Gold):** Experiment 1's published p-values depend slightly on how tied earthquakes of equal magnitude are
ordered when aftershocks are removed (the committed script reproduces the committed file in 13 of 46 rows, but the published value
lies inside the range of 200 random orders in 45 of 46, and no order ever passes the bar; smallest p over all runs 0.0188). Experiment 2
stage 5 reproduces exactly. See `gold_experiment1/README.md` and `gold_experiment2/README.md`.

## 2. Adding a new treatise: steps from Bronze to a re-run experiment

Run everything from the repository root. Install once: `python3 -m pip install -r requirements.txt` and `brew install poppler tesseract tesseract-lang`.

**Bronze: register and extract the book**
1. Pick the next ID (`TXT-14`, ...) and add the book to `TEXTS` in `bronze_text_sources/pipeline/build_register.py` (and, optionally, a file-name hint in `suggest_raw_index.py`).
2. Copy the PDF into `bronze_text_sources/raw/` (never committed; keep your own backup).
3. `python3 bronze_text_sources/pipeline/suggest_raw_index.py`, then open `raw/index.csv` and set the new file's `text_id` (file names carry no meaning).
4. `python3 bronze_text_sources/pipeline/register_raw_batch.py` (fingerprints the file and appends to the audit chain), then `python3 bronze_text_sources/pipeline/build_register.py`.
5. `python3 bronze_text_sources/pipeline/extract_text.py --plan`, then `extract_text.py --only RAW-xxxxxxxxxxxx --jobs 2` for the new file (safe to stop and re-run; about 25 OCR pages a minute).
6. `python3 bronze_text_sources/pipeline/verify_extracts.py RAW-xxxxxxxxxxxx` must print "All checks passed". Look at the low-confidence pages before relying on the text.

**Silver: turn the text into testable rules**
7. Write the rules the book supports as testable claims in `silver_text_rules/rules/<name>_rules.json` (same shape as `experiment1_rules.json`; set `text_id`, the source reference, the exact test and its parameters). They are checked against `schema/silver_rule.schema.json`.
8. **Pre-register before testing:** add `docs/audit/preregistrations/preregistration_<name>.md` fixing the number of tests (this sets the Bonferroni bar, 0.05 ÷ tests), the discovery/hold-out split and the pass rule. Commit it before step 10.
9. `python3 silver_text_rules/pipeline/build_silver.py` rebuilds `silver.duckdb`. It stops if the audit chain, a hash or a rule fails its check. Every `*_rules.json` is picked up automatically; a `<name>_reported_results.csv` beside it is optional until results exist.

**Gold: run and check**
10. Copy `gold_experiment2/pipeline/run_stage5.py` as the template for the new study (it reads the rules and published results from Silver, runs the analysis script unmodified, and compares). Generate inputs with the scripts in `analysis/shared/`.
11. Run it once. Save the numbers as `<name>_reported_results.csv`, rebuild Silver (step 9), and from then on re-runs are checked against those frozen results.

**Notes**
- Experiments 1 and 2 stay frozen: do not edit their rules or scripts. A new treatise is a **new** pre-registered batch with its own Bonferroni family. If you also want the old and new rules judged together, say so in the new pre-registration and report both views.
- Run `verify_extracts.py` and `build_silver.py` again after any change; they are the gates between layers.
- Claude does not push to git. Commit the new rules, pre-registration and Gold script yourself; scans, `extracted/` and `*.duckdb` stay out of git.
