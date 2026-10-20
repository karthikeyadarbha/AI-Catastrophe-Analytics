# Pre-registration: Vedic earthquake-pattern rules (frozen 2026-10-20, before any test on the holdout set)

## Data
- Events: USGS ComCat, worldwide, M>=6.5, 1990-01-01 to 2026-10-19 (1,709 events).
- Declustering: Gardner-Knopoff style window (time 10^(0.032M+2.7389) days, distance 10^(0.1238M+0.983) km); keep the largest event in each window.
- Split: DISCOVERY = events before 2008-01-01; HOLDOUT = events from 2008-01-01. The holdout is the confirmatory set and each rule is run once on it.
- Positions: astral-ephemeris tool, Lahiri sidereal and tropical (sayana). Eclipses: own Meeus finder (+/-1.4 d).
- Null: expected share = share of a uniform 6-hourly background grid over 1990-2026 (or the analytic share for equal-width classes).

## Rules (each is one test; chi-square for multi-class, upper-tail binomial for single-class)
Mandalas (corrected from the Sanskrit of 18.3-18.6; Abhijit excluded):
- Anila: Svati, Dhanishta, Ashwini, U.Phalguni, Hasta, Chitra
- Anala: Krittika, Vishakha, P.Bhadrapada, Bharani, P.Ashadha, Magha, Pushya, P.Phalguni
- Indra: Jyeshtha, U.Ashadha, Shravana, Rohini, Mrigashira, Punarvasu, Anuradha
- Varuna: Shatabhisha, U.Bhadrapada, Revati, Ardra, Ashlesha, Mula

1. R1 Mandala of Moon nakshatra (4 classes, chi-square) - sidereal and tropical (2 tests)
2. R2 Moon in sharp group (Ardra, Ashlesha, Jyeshtha, Mula) - sidereal and tropical (2 tests)
3. R3 Moon in Rikta tithi (tithi 4, 9, 14 of each 15-cycle; share 1/5) (1 test)
4. R4 Moon within +/-1 tithi of new or full moon (share 6/30) (1 test)
5. R5 Eclipse within +/-15 days (1 test)
6. R6 Eclipse within +/-7 days (1 test)
7. R7 Mars retrograde; Mercury retrograde; Jupiter retrograde; Saturn retrograde; Venus retrograde (5 tests)
8. R8 Jupiter and Saturn in the same nakshatra - sidereal and tropical (2 tests)
9. R9 Any pair of Mars, Mercury, Jupiter, Venus, Saturn within 3 degrees (1 test)
10. R10 Moon within 5 degrees of a planet or the Sun (1 test)
11. R11 PT season (6 equal 60-degree seasons from the winter solstice / Dhanishtha start), chi-square - sidereal and tropical (2 tests)
12. R12 Mandala vs magnitude (Kruskal-Wallis) and mandala vs depth (Kruskal-Wallis), sidereal and tropical (4 tests)

Family size m = 23 tests; Bonferroni threshold on the holdout = 0.05/23 = 0.00217.

## Interpretation rule (fixed in advance)
A pattern counts as supported only if it passes the Bonferroni threshold on the holdout AND has the same direction in the discovery set. Otherwise it is reported as not supported.
