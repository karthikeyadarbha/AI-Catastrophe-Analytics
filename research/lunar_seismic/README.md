# Lunar tidal triggering of earthquakes — a controlled study

Does the Moon's local tidal geometry modulate when earthquakes happen? This
package tests the hypothesis in physical terms and — crucially — against a
matched random-time null with multiple-testing correction, so exploratory
sweeps don't report chance as discovery.

## The hypothesis, operationalized

Original idea: *"the Moon's declination at its peak, with other bodies pulling
in other directions, triggers earthquakes when the Moon is rising/setting or
exactly overhead at that place."* This is the **lunar tidal-triggering**
hypothesis. Each phrase maps to a measurable, per-earthquake quantity computed
at the event's own epicenter and UTC time:

| Phrase | Variable (`geometry.py`) | Physics |
|---|---|---|
| "moonrise / moonset" | `moon_hour_angle_h` ≈ ±6ʰ (altitude ≈ 0°) | max horizontal tidal shear |
| "Moon exactly at the top" | hour angle ≈ 0ʰ (altitude ≈ +90°) | max vertical tide (sub-lunar bulge) |
| "declination at its peak" | `abs_moon_dec` near standstill | tidal bulge at high latitude |
| "other bodies pulling" | `sun_moon_elong_deg`, `tide_vertical` | Sun = 46% of Moon's tide |
| perigee (implied) | `moon_dist_km` | tide ∝ 1/distance³ |

**Physics refinement:** the solid-Earth tide has *two* daily maxima — Moon
overhead **and** underfoot — so "at the top" is tested as hour angle ≈ 0ʰ *or*
±12ʰ via the **semidiurnal** phase (hour angle × 2). The `tide_vertical` feature
encodes both bulges directly through the (3cos²z − 1) tidal-potential term.

Honest caveat: the **planets'** tidal contribution is ~10⁻⁵ of the Moon's, so
the only "other body" that matters physically is the Sun.

## Method

```
USGS ComCat  →  Gardner–Knopoff decluster  →  per-event lunar geometry
   (catalog.py)      (decluster.py)               (geometry.py, Skyfield/DE421)
        →  random-time null  →  Schuster + Monte-Carlo tests  →  BH-FDR
              (nulls.py)                (stats.py)              (pipeline.py)
```

1. **Catalog** — global events from the USGS FDSN service, fetched in yearly
   chunks and cached.
2. **Declustering** — Gardner & Knopoff (1974) space-time windows remove
   foreshocks/aftershocks. This is essential: an aftershock swarm shares one
   sky and would otherwise fake a huge correlation.
3. **Geometry** — vectorized *geocentric* lunar/solar hour angle, altitude,
   declination, distance, sub-lunar point and a combined Sun+Moon vertical-tide
   scalar, via Skyfield over a JPL DE ephemeris.
4. **Null model** — hold each epicenter fixed, draw *k* random times across the
   catalog span, recompute geometry. This captures the real (non-uniform)
   distribution the features take by chance.
5. **Tests** — Schuster's test for phase clustering (the standard tidal-triggering
   tool); Monte-Carlo one-sided tests for the continuous features. All p-values
   are corrected with Benjamini–Hochberg FDR. Analysis is stratified by depth
   (all vs. shallow ≤ 70 km, where tides can plausibly matter).

## Results (global M ≥ 6.0, 1973–2024)

7,182 events → **1,968 independent mainshocks** (1,564 shallow). Random-time
null with k = 200 replicates.

| Test (all depths) | effect | raw p | FDR q | significant? |
|---|---|---|---|---|
| Schuster diurnal (hour angle) | r̄ = 0.028 | 0.21 | 0.75 | no |
| Schuster semidiurnal (top/bottom vs horizon) | r̄ = 0.006 | 0.93 | 0.94 | no |
| Higher vertical tide | z = −0.90 | 0.82 | 0.94 | no |
| **Larger \|lunar declination\|** | **z = +2.52** | **0.020** | 0.28 | no |
| Smaller Moon distance (perigee) | z = +0.56 | 0.72 | 0.94 | no |
| Nearer syzygy (spring tide) | z = −0.31 | 0.63 | 0.94 | no |
| Higher raw tidal strength | z = −0.45 | 0.70 | 0.94 | no |

**Conclusion: no robust lunar tidal-triggering signal at M ≥ 6 globally.** The
lunar hour angle at earthquakes is statistically flat (see
`hour_angle_all_depths.png`) — no pile-up at moonrise/set or at culmination.
The one nominal hint is a **weak excess of high lunar declination** (raw
p ≈ 0.02), but it does **not** survive multiple-testing correction (q ≈ 0.28),
and across 14 tests a raw p ≈ 0.02 is expected roughly once by chance. So it is
a lead to probe, not a finding.

This negative result is consistent with the geophysics literature: robust tidal
effects appear mainly (i) for the tidal **shear stress resolved on the fault
plane** (needs focal mechanisms), (ii) in specific tectonic settings (shallow
submarine thrusts), or (iii) for the very largest events near failure — not in a
bulk global catalog tested against the raw tidal potential.

## Limitations & next steps

- **Fault-resolved stress**: the strongest published signals use tidal Coulomb
  stress on each fault's geometry (from GCMT moment tensors), not the scalar
  potential used here. This is the highest-value extension.
- **Magnitude / region sweeps**: repeat for M ≥ 7 (cf. Ide et al. 2016) and for
  individual subduction zones, with FDR across the family.
- **Declination follow-up**: the weak `abs_moon_dec` lead deserves a
  pre-registered, single-hypothesis test on an independent time span.

## The exploratory sidereal ("astrology") battery

Beyond the physically-motivated tidal test above, we ran a deliberately broad
sweep of **sidereal / Vedic** features to check whether *any* classical
astrological configuration is over-represented at earthquakes. This has **no
known physical mechanism** — it is a pure pattern search, run honestly with the
same guards (matched null + multiple-testing correction) so a chance pattern
cannot masquerade as a discovery.

The battery now runs on the reusable **`astro_engine.vedic`** library: a
vectorized substrate (`SkySample`) feeds eleven use-case modules, each emitting a
uniform `FeatureSet` of categorical + boolean features, and
`research/lunar_seismic/vedic_battery.py` tests *every* feature generically —
chi-square for each categorical, a binomial for each flag — against the matched
null, then BH-FDR-corrects the lot.

**Features tested (724 tests total, all depths + shallow ≤ 70 km):**

| Family | Examples | Test |
|---|---|---|
| `signs` | sidereal rasi of the 9 grahas + ascendant | χ² (12-way) |
| `panchanga` | tithi, karana, yoga, Moon nakshatra + pada, vara | χ² |
| `bhava` | whole-sign house of each graha; dusthana | χ² / binomial |
| `varga` | Navamsa (D9) sign of each graha | χ² (12-way) |
| `dasha` | running Vimshottari mahadasha lord | χ² (9-way) |
| `dignity` | exalt/own/friend/…/debilitated, combustion, graha yuddha, stationary | χ² / binomial |
| `aspects` | Ptolemaic aspects, Vedic drishti, **declination parallels** | binomial |
| `declination` | **out-of-bounds**, hemisphere, nodal standstills | binomial |
| `cycles` | slow outer-planet pair phases (mundane astrology) | binomial |
| `stars` | Sun/Moon/Lagna on prominent fixed stars | binomial |
| `upagraha` | Gulika/Mandi sign, day/night | χ² / binomial |

**Null model.** For each mainshock the epicenter is held fixed and *k* = 100
random times are drawn across the catalog span; the pooled category frequencies
of those ~197k random skies (per stratum) define the expected distribution.
**Sun's sign (= time of year) is a negative control**: the null already knows the
observing times, so a correct pipeline must return Sun-sign as non-significant.

| Test | stratum | effect | raw p | FDR q | significant? |
|---|---|---|---|---|---|
| Uranus–Neptune conjunction | all | ratio 1.25 | 0.000095 | 0.069 | no |
| Jupiter–Uranus square | shallow | ratio 1.40 | 0.0009 | 0.20 | no |
| Karana (enriched: Chatushpada) | shallow | V = 0.043 | 0.0011 | 0.20 | no |
| Rahu / Ketu sign (Capricorn/Cancer) | all | V = 0.035 | 0.0044 | 0.36 | no |
| Moon out-of-bounds (declination lead) | all | ratio 1.14 | 0.014 | 0.62 | no |
| **Sun sign (negative control)** | all | V = 0.019 | **0.625** | 0.96 | no |

**Conclusion: nothing survives.** Across 724 sidereal tests the smallest raw
p ≈ 9.5 × 10⁻⁵ (Uranus–Neptune conjunction) gives an FDR q ≈ 0.069 — the closest
any feature comes, and it fails. That "hit" is in any case a **slow-cycle
artifact**: the Uranus–Neptune conjunction was a single multi-year epoch (~1993),
so its "within orb" flag is really a coarse decade indicator, not 1968 independent
draws — exactly the kind of low-effective-DoF nuisance a mechanism-free scan
throws up. Effect sizes everywhere are negligible (Cramér's V ≈ 0.03–0.04; rate
ratios ≈ 1.2–1.4 on rare flags). The declination out-of-bounds flag reappears in
the *same* direction as the tidal study's one weak lead (ratio 1.14) but again
does not survive. The Sun-sign control lands at p = 0.63 / q = 0.96, confirming
the null model neither leaks a seasonal artifact nor manufactures false positives.

*(Rahu/Ketu, node-aspects and some declination pairs appear as duplicated rows
because Ketu ≡ Rahu + 180°; BH-FDR is conservative under this positive
dependence, so it does not affect the null conclusion. Expanding from 230 to 724
features lowers per-feature power — the price of breadth — but BH still controls
the false-discovery rate, and nothing clears it.)*

**What a real hit would require.** Any q < 0.05 survivor here would still only be
a *candidate* — it would need **out-of-sample replication** (e.g. fit on
1973–1999, confirm on 2000–2024) and a plausible mechanism before it counted as a
correlation, let alone a step toward causation. No candidate reached even that
first bar.

## Running it

```bash
# tidal study
python -m research.lunar_seismic.pipeline       --kernel path/to/de421.bsp --minmag 6.0 --k 200
# full sidereal battery (all 11 vedic modules)
python -m research.lunar_seismic.vedic_battery   --kernel path/to/de421.bsp --minmag 6.0 --k 100
# focus a single hypothesis (sharper power, fewer comparisons)
python -m research.lunar_seismic.vedic_battery   --include declination aspects --minmag 6.0 --k 100
```

Outputs land in `research/lunar_seismic/outputs/` (`results.csv`,
`vedic_results.csv`, plus diagnostic plots). Data/outputs are git-ignored. The
earlier hand-listed battery (`astro_pipeline.py`, 230 tests) is superseded by
`vedic_battery.py` but kept for reference.

### References
- Gardner & Knopoff (1974), *BSSA* — declustering windows.
- Schuster (1897) — test for tidal periodicity in earthquakes.
- Cochran, Vidale & Tanaka (2004), *Science* — tidal triggering of shallow thrusts.
- Métivier et al. (2009), *EPSL*; Ide, Yabe & Tanaka (2016), *Nature Geoscience*.
