# Pre-registration: TG-SAI out-of-sample test and data-first scan on historical earthquakes
Written 20 Oct 2026 before any historical event was scored by the model or scanned. Follows preregistration_historical_test.md (same event sets: Era A 1964-1989 M>=5.5, declustered, 6,111 events; Era B 1900-1963 M>=7.0, 670 events).

## Part 1: TG-SAI (RandomForest) out-of-sample
- Training (once, never touched afterwards): all 9,368 declustered M>=5.5 events of 1990-2026 as positives; an equal number of negatives = random times from the 1990-2026 six-hourly background, with latitudes shuffled from the event latitudes (matched latitude distribution). RandomForest: 300 trees, min_samples_leaf=5, seed 0.
- Models (the three used before): A = the repository's seven features computed with its own formulas (crude planet positions); B = seven features with true geocentric planet positions; C = B plus Moon-longitude, tithi and mandala features.
- Test sets, no retraining: Era A and Era B events as positives; negatives = random times from each era's own six-hourly background with event latitudes shuffled; 10 independent negative draws per era, AUC averaged.
- Primary: Model A on Era A. Passes only if mean AUC >= 0.52 and the 95% bootstrap interval (1,000 resamples of events) has lower bound > 0.5.
- Secondary: B and C on Era A (alpha 0.05/3 by shuffled-label permutation test, 200 permutations); all three on Era B (descriptive, low power); all three on the repository's own 1850-1950 M5+ file (ISC-GEM based; declustered, M>=5.5; overlaps Era B for 1900-1950, so not independent of it).
- A model that passes primary AND secondary on B or C is the only outcome that counts as evidence the sky features carry any signal.

## Part 2: data-first scan repeated
- Same 433 bins as the 1990-2026 scan. Compute per-bin z scores (events vs period-matched background) for Era A.
- Statistic: Pearson correlation between the Era A z-vector and the 1990-2026 (all 9,368 events) z-vector. Null: 300 resamples of random background times with the same count. Passes if p < 0.05 (upper tail).
- Also list the 5 largest 1990-2026 bins and whether any reaches |z|>2.5 with the same sign in Era A (descriptive).
## Decision rule
Anything not meeting its stated threshold is reported as no evidence. No threshold or feature is changed after seeing results.
