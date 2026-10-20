# Pre-registration: Brihat Samhita English cross-check and re-execution
Written 20 Oct 2026, before any test below was run. Source: "The Brihat Samhita of Varaha Mihira" (English translation, uploaded PDF) compared with my own reading of the Sanskrit scan.

## Comparison findings (read before testing)
- Ch.32 v.8, 12, 16: the Vayu, Agni, Indra circles in the English match the Sanskrit reading exactly (7 nakshatras each, Abhijit in Indra).
- Ch.32 v.20: the English prints U.Ashadha in BOTH the Indra circle (v.16) and the Varuna circle (v.20) and omits U.Bhadrapada. The Sanskrit reading has U.Bhadrapada in Varuna. The English is therefore treated as a printing slip; the Sanskrit list stays primary and the English-as-printed lists are run as sensitivity variants.
- Ch.32 v.7: four circles act "in the first, second, third and fourth six hours of day and night": consistent with the Sanskrit reading (Vayu, Agni by day, Indra, Varuna by night) if each is a six-hour quarter of the day counted from sunrise.
- Ch.32 v.34: a second earthquake on the 3rd, 4th, 7th, 15th, 30th or 45th day: the "15" reading is supported (my earlier robustness grid already tested both 14 and 15; no change).
- New testable statements found in the English: v.27 (time of day overrides nakshatra circle), v.28-31 (mixed circles), v.33 (felt distance: Vayu 200 yojanas, Varuna 180, Indra 160, Agni 110), ch.5 v.92 (an earthquake within seven days after an eclipse ends).
- Not testable: effects on people, crops, kings, countries; v.32 durations of other portents.

## Tests (family of 10 per data set; Bonferroni 0.05/10 = 0.005)
Data sets: 1990-2026 M>=5.5 declustered (9,368; reported all, before 2008 and 2008+), Era A 1964-1989 (6,111), Era B 1900-1963 M>=7 (670). Backgrounds: period-matched six-hourly grids, event longitudes shuffled onto background times for any local-time rule. Lahiri sidereal (tropical not repeated).
1-4. Mandala by Moon nakshatra, chi-square vs background: (1) Sanskrit lists; (2) English as printed with U.Ashadha in Indra, U.Bhadrapada unassigned; (3) English with U.Ashadha in Varuna, U.Bhadrapada unassigned; (4) U.Ashadha and U.Bhadrapada both set aside.
5. Time-of-day circle (local mean solar time quarters: Vayu 06-12, Agni 12-18, Indra 18-24, Varuna 00-06), chi-square. This also tests the v.27 precedence rule, which makes the time circle the effective circle.
6. Agreement of nakshatra circle and time circle, binomial upper tail.
7. All 16 (nakshatra circle, time circle) pairs, chi-square (v.28-31 mixed circles).
8-9. Felt-distance ordering (Vayu > Varuna > Indra > Agni): one-sided Spearman correlation of magnitude with the assigned distance, by nakshatra circle (8) and by time circle (9).
10. Eclipse within 7 days before the earthquake (v.92), binomial upper tail vs background; eclipse dates by the Meeus mean-phase finder (about +/-0.6 day).
## Decision rule
A rule is supported only if p < 0.005 in the 1990-2026 2008+ holdout, shows the same direction (p < 0.05) in Era A, and is not explained by a time-shift check. Otherwise: no evidence.
