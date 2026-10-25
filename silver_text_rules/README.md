# Silver layer: text rules

Silver turns Bronze's verified page text into something that can be queried and reviewed.
It never changes Bronze or the frozen engine (`indic-ganita-engine` v1.4.2).

## Build
```
python3 -m pip install duckdb
python3 silver_text_rules/pipeline/build_silver.py
```
Writes `silver_text_rules/silver.duckdb` (git-ignored, rebuilt from scratch each run).
The build refuses to start unless the audit chain verifies and every extract's page files
add up to the hash the audit log holds, so the database cannot hold text that Bronze
does not vouch for.

## Tables
| Table | Rows | What it is |
|---|---|---|
| `texts` | 13 | The register of classical texts |
| `artifacts` | registered scans | File, hash, type |
| `extracts` | one per scan | Tools, languages, OCR quality summary |
| `pages` | one per page | Text, method (text layer or OCR), confidence, flag, page hash |
| `rules` | 85 | The engine's rule records, read-only, with `rule_kind = 'computational'`, `provenance_status` and `review_status` |
| `rule_sources` | per rule page | Page references exactly as the engine records give them |

## Two kinds of rule
- **computational**: how a text calculates something (revolutions in a yuga, Rsine table). The engine's 85 records are this kind.
- **testable_claim**: a statement that can be tested against earthquakes (for example the Moon's mandala at the time of the event). Experiment 1 uses these. They are not in the table yet; the 23 pre-registered rules in `docs/audit/preregistrations/preregistration_rules.md` are the first to encode.

Both kinds share one record shape: `schema/silver_rule.schema.json`. Engine IDs are kept as they are (the old Bronze pattern `^[A-Z]{2,3}-[0-9]{3,4}$` did not fit them).

## Path to Gold (Experiment 1)
1. Encode each pre-registered rule as a `testable_claim` with its condition, parameters and the page it comes from.
2. A person reviews it (`review.status = approved`). Only approved rules reach Gold.
3. Gold joins approved rules with the earthquake catalogue and the matched background grid, and runs the same statistics as today, one results row per rule ID.
4. First check: the Gold run must reproduce the existing result CSVs exactly. New rules need a new pre-registration.

## Known limits
- 67 of the 85 engine rules cite a reference but no PDF page, so they cannot yet be linked to a scanned page.
- Page text is raw OCR. The weakest are Adbhuta Sāgara (mean 65.7%) and the unscored text layer of the English Bṛhat Saṃhitā. Review is where this is corrected.
- IAST conversion is not done (Aksharamukha is not wired in).
