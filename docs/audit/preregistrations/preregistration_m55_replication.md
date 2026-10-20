# Pre-registration: replication on the larger M5.5+ catalogue (written 2026-10-20 before any M5.5-6.4 event was analysed)

Data: USGS worldwide earthquakes M5.5 and above, 1990-01-01 to 2026-10-19, declustered with Gardner-Knopoff windows (time window 10^(0.5409M-0.547) days for M<6.5, 10^(0.032M+2.7389) for M>=6.5; distance 10^(0.1238M+0.983) km). The 1,270 M>=6.5 events already analysed are kept; the NEW events are those with M 5.5-6.49.
Sky positions: computed locally with an own ephemeris (Meeus Sun and Moon series, Standish planetary elements), validated against the astral-ephemeris tool on the 1,270 events (median difference 0.006-0.08 deg, maximum 0.17 deg; retrograde flags 100% identical). Lahiri ayanamsa. Expected counts come from a 6-hourly background grid 1990-2026, matched to the years analysed.

## Primary confirmatory tests (on NEW events only, M5.5-6.49; each one-sided, p < 0.05 / 2 = 0.025)
P1. Moon in the sharp nakshatra group (Ardra, Ashlesha, Jyeshtha, Mula), sidereal Lahiri. Excess expected.
P2. Sun in Vishakha (sidereal longitude 200.00-213.33 deg). Excess expected.

## Secondary
S1. The 23 rules of preregistration_rules.md rerun on all events M5.5+ (discovery before 2008, holdout 2008 onward), background matched to each half; threshold 0.05/23 = 0.00217.
S2. The bottom-up scan (433 bins), discovery vs holdout correlation r, with 300 resamples for chance, on all events M5.5+ and on new events only.
S3. Mandalas (Parashara Tantra and Brihat Samhita) by Moon nakshatra, chi-square, new events only.

Verdict rule: a primary test counts as confirmed only if p < 0.025 in the stated direction.
