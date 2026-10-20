# Pre-registration: Adbhuta Sagara (Vallala Sena, ed. Murali Dhara Jha, Varanasi 1905), chapter "Bhumikampadbhutavartta" (printed pp. 383-408)
Written 20 October 2026, after reading the chapter from the page images and before any test below was run.

## What the chapter is
A compilation of quotations on earthquakes from Vishnu Purana, Kashyapa, Garga, Vriddha-Garga, Vasishtha, Parashara (as quoted), Brihaspati, Bhargava (Shukra), Ushanas, Atharvana-adbhuta, Mayuracitra, Vatakanika, Matsya Purana, Vishnudharmottara and the Brihat Samhita, with five-fold shanti rites. It uses the same four circles (Vayu, Agni, Indra, Varuna) and adds authority-specific star lists, a day-part (yama) clock, twilight and distance rules.

## Readings fixed in advance (nakshatra index: Ashwini 0 ... Revati 26)
* PT-as-quoted (Parashara as cited): Vayu [14,22,0,11,12,13]; Agni [2,15,24,1,19,9,7,10]; Indra [17,20,21,3,4,6,16]; Varuna [23,25,26,5,8,18] (identical to the corrected Parashara Tantra lists of the Rule-First Study, so this is a check of that correction).
* Brihaspati: Vayu [11,12,13,14,4,0] (Mrigashira in place of Dhanishtha; Abhijit extra); Agni [7,2,15,1,9,24,10]; Indra [17,16,21,22,6,3,20] (Jyeshtha, Anuradha, Shravana, Dhanishtha, Punarvasu, Rohini, U.Ashadha); Varuna [26,19,5,8,18,25,23].
* Compiler's own summary (p.404): Vayu [11,12,13,14,6,4,0]; Agni [10,7,2,15,1,9] (P.Bhadrapada not listed); Indra [3,17,20,16,21,22]; Varuna [26,19,5,8,18,25,23].
* Disputed nakshatras (placed in different circles by different authorities): Mrigashira 4, Punarvasu 6, P.Ashadha 19, Dhanishtha 22.
* Yama clock (Atharvana-adbhuta, Matsya, Vishnudharmottara): the daytime from sunrise to sunset is split in four equal yamas giving Vayu, Agni, Indra, Varuna; the night from sunset to sunrise is split in four, in reverse order (Varuna, Indra, Agni, Vayu). Local solar time, sunrise and sunset from latitude and the Sun's declination (equation of time ignored).
* Twilight (Mayuracitra: a quake at the two sandhyas is an ill omen): within 48 minutes of local sunrise or sunset.
* Felt distance, Bhargava order: Vayu 120 > Agni 90 > Indra 80 > Varuna 70 (Brihat Samhita order 200/180/160/110 was tested earlier).

## Tests (family of 9; Bonferroni bar 0.05/9 = 0.0056)
T1 four-circle chi-square, Brihaspati lists. T2 same, compiler's summary lists (events in P.Bhadrapada excluded). T3 same, PT-as-quoted lists. T4 Moon in a disputed nakshatra (binomial, upper tail). T5 yama-clock circle, four-class chi-square. T6 yama-clock circle equals nakshatra circle (BS lists). T7 same, Brihaspati lists. T8 twilight window (binomial). T9 Spearman correlation of Bhargava felt distance (nakshatra circle, BS lists) with magnitude, one-sided.

## Data and background
Modern M5.5+ set (all years, 1990-2007 discovery, 2008-2026 holdout), Older set 1 (1964-1989), Older set 2 (1900-1963). Background: the same 6-hourly sky grids as before with event locations shuffled onto background times. The Deadly 15 are listed descriptively. A result counts only if it passes 0.0056 in the 2008-2026 holdout AND points the same way in the 1990-2007 discovery half. Results in the older sets are replication checks.

## Aftershock lags
Kashyapa's verse ("half-month, fourth day, month and a half, seventh") and Parashara's quoted line give the same lag set already tested (3, 4, 7, 15, 30, 45 days); no new test.
## Not testable
Omens (winds, meteors, cloud colours), rain and crop results, direction in which the tremor travels, effects on kings and peoples, the 7-day/1.5-month/3-month fruition times, parvan versus non-parvan quality, and the five-fold shanti rites.

## Amendment (20 October 2026, after the first run)
The first run used the unshifted 6-hourly background grid. Its 6-hour spacing aligns with the 6-hour yama pattern and produced a spurious twilight deficit (0.89x) and a spurious T5 hit in Older set 1 (p=0.012). Background times were then given a random 0-6 h shift (seed 11) and all tests re-run; the reported results are the shifted ones. The test list, lists and bar were not changed.

## Amendment 2 (21 October 2026, after reading every Parashara-attributed quotation)
Reading all Parashara quotations showed (a) his Agni list as quoted omits P.Ashadha, and (b) Matsya Purana / Vishnudharmottara put night yamas in the SAME order as day (Vayu, Agni, Indra, Varuna), while the Atharvana-adbhuta reverses them (the only variant in T5-T7). Four tests were added: T3b (Parashara lists exactly as quoted, P.Ashadha unlisted), T5b (same-order yama circle), T6b and T7b (same-order yama circle equals nakshatra circle, BS and Brihaspati lists). The family is now 13 tests, bar 0.05/13 = 0.0038, applied to all tests. Result: none passes (smallest p = 0.072).
