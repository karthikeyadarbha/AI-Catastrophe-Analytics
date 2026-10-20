# Pre-registration: historical (pre-1990) replication
Written 20 Oct 2026, before any historical earthquake was run through any test. Only the event counts were looked at.

## Data
- Source: USGS ComCat (includes the reviewed ISC-GEM catalogue for pre-1990 events). NOAA NCEI and ISC sites are not reachable from the workspace.
- Era A (primary): 1964-01-01 to 1989-12-31, M>=5.5 (ISC-GEM is designed to be complete at M5.5 from 1964).
- Era B (secondary, low power): 1900-01-01 to 1963-12-31, M>=7.0 (complete for large events only).
- Declustering: Gardner-Knopoff exactly as in the 1990-2026 study (same windows).
- Sky positions: own ephemeris (validated, max diff 0.17 deg), Lahiri ayanamsa; planetary elements are valid 1800-2050, so both eras are inside the valid window. Background: 6-hourly grid over each era, same code.
- Eclipse rules are NOT run (eclipse catalogue not built for these years); stated as a gap.

## Primary hypothesis (one test, alpha = 0.05, one-sided upper tail)
H1: In Era A, among events of M5.5-5.9, the Moon lies within 5 deg (ecliptic longitude) of the Sun or of Mars, Mercury, Jupiter, Venus or Saturn more often than the period-matched background. Binomial upper-tail test. Confirmed only if p < 0.05 AND ratio > 1.

## Secondary tests (reported whatever the outcome)
S1. Same rule, Era A all M>=5.5; Era A M>=6.0.
S2. Same rule, Era B (M>=7.0), descriptive.
S3. Replication family: all 23 rules from the 1990-2026 pre-registration, run on Era A with the same definitions (frames: sidereal and tropical where originally both). Bonferroni threshold 0.05/23 = 0.00217. Eclipse rules R5, R6 excluded (see above) so the family is 21 rules, threshold 0.05/21 = 0.00238.
S4. The two earlier leads: Moon in the sharp nakshatra group (Ardra, Aslesha, Jyeshtha, Mula) and Sun in Vishakha; threshold 0.025 each.
S5. Time-shift null for H1: shift all event dates by a random whole number of days in [-3650, 3650] (500 draws) and compare the observed ratio with the shifted distribution.

## Decision rule
- H1 "replicated" only if primary p<0.05 and time-shift p<0.05.
- If H1 fails, the Moon-near-Sun-or-planet lead is closed as unsupported.
- No other result may be called a finding unless it passes its stated threshold; anything else is descriptive.
## Known limits (stated in advance)
Older magnitudes are less consistent; time stamps are UTC seconds-level for ISC-GEM events so Moon-based rules are unaffected; place and depth errors do not matter for sky rules. Era B has about 1/10 of the events, so power is low.
