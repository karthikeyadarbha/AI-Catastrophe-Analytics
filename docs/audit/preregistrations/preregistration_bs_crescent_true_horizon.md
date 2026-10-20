# Pre-registration: true-horizon crescent test (Brihat Samhita ch.4 v.12)
Written 20 Oct 2026, before any computation of the crescent condition for any earthquake.
Verse (English translation supplied): "If the moon should appear stretched from North to South presenting the appearance of a carriage pole, there will be earthquake (within that month)." Footnote: the observation is made on the first or second lunar day.

## Quantities
- Month: the lunar month (new moon to next new moon) containing the earthquake. New-moon time from the ephemeris (own Meeus series; Moon latitude series validated against Meeus's worked example to 0.00001 deg).
- Sighting: on each of the first four evenings after the new moon, the first moment the Sun is 3 deg below the horizon; the first evening on which the Moon is then at least 3 deg above the horizon and at least 8 deg from the Sun is the sighting. No sighting (e.g. polar latitudes) = the month is excluded.
- Horn-line tilt: angle between the line joining the horns (perpendicular to the Sun's direction as seen from the Moon) and the horizon. 0 = level (both horns at the same height).
- "Yoke month" (primary condition): tilt <= 20 deg AND Moon azimuth within 30 deg of due west or east, so that the level horn line runs north-south. Secondary condition: tilt <= 20 deg only.
- No parallax or refraction correction (both are about 1 deg, far inside the 20 and 30 deg tolerances).

## Tests (family of 4 per data set; Bonferroni 0.05/4 = 0.0125)
Observer location is not stated in the verse, so two readings are fixed in advance:
- U1 (primary): observer at Ujjain (23.18 N, 75.78 E), the traditional meridian of the text; condition = yoke. Because many earthquakes share a month, the p-value is from a date-shift null (500 shifts of all event dates by a whole number of days within +/-3650); upper tail.
- U2: same, tilt-only condition.
- L1: observer at the epicentre; condition = yoke; binomial upper tail against the rate for random times with shuffled event locations (3 draws).
- L2: same, tilt-only condition.
Data sets: 1990-2026 M>=5.5 declustered (9,368; all and 2008+ holdout), Era A 1964-1989 (6,111), Era B 1900-1963 M>=7 (670).
## Decision rule
Supported only if p < 0.0125 in the 2008+ holdout AND p < 0.05 in the same direction in Era A. Otherwise, no evidence. Thresholds, tolerances and definitions are not changed after seeing results.
