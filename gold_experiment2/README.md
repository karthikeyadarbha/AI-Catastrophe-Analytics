# Gold: Experiment 2 (Data-First), stage 5

Stage 5 scans the 54 bins that the source texts themselves define (Moon circles under eight lists,
yama-clock circles, clock-time circle, three single-condition bins, three planetary-war bins). The
bins live in Silver as rules `S5.*` (`silver_text_rules/rules/experiment2_stage5_rules.json`); this
folder re-runs the scan and checks it against Silver and the committed results.

## Run it
```
python3 silver_text_rules/pipeline/build_silver.py        # once, makes silver.duckdb
python3 gold_experiment2/pipeline/run_stage5.py --inputs analysis/workdir --null
```
`--inputs` holds `quakes_m55_input.csv`, `bg_sid.csv`, `histA_declustered.csv`, `histB_declustered.csv`;
`m55_features_sid.csv` is made from them by `analysis/shared/big55.py` if it is missing (written into
that input folder, which is git-ignored). `--null` also reproduces the cross-era agreement test
(about a minute). Results go to `gold_experiment2/results/` (git-ignored).

## What is checked (all hard)
- A. Silver's published p-values, hold-out counts and expected counts equal the committed CSV.
- B. The re-run reproduces the committed CSV exactly: counts identical, z-scores within 1e-9, in all four periods.
- C. The pass rule (p < 0.05/54 = 0.00093 in 1990-2007 and in 2008-2026, same direction) gives the same findings: none.
- D. With `--null`: the agreement test (3,000 redraws, fixed seed) matches the committed file.

Unlike Experiment 1, this stage starts from files that were declustered earlier, and the random
jitter uses fixed seeds, so an exact match is expected and was obtained (largest difference 0).
The tie-order question of Experiment 1 would sit one step upstream, in how those input files were
made; stages 1-4 and the M5.5+ declustering itself are not covered here yet.

## Not yet done
- Stages 1 to 4 (generic bins, M5.5+ replication of the 23 rules, gap tests, older eras) have no text-defined rules to store in Silver; they can be re-run and compared the same way.
- Which text defines the three single-condition bins is not stated in the report, so their `text_id` is empty in Silver.
- The 54 bins are `unreviewed` and cite sources at chapter level only.
