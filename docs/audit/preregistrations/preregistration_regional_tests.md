# Pre-registration: regional (belt-by-belt) tests
Written 20 Oct 2026, before any rule was evaluated within any region. Only the number of events per region may be looked at before testing.

## Regions (fixed boxes, first match wins in this order; events outside all boxes form "Rest of world", reported but not in the test family)
1. Japan-Kurils-Korea: lat 30 to 52, lon 128 to 160
2. Philippines-Taiwan: lat 6 to 26, lon 116 to 128
3. Indonesia (Sunda arc to Papua): lat -12 to 6, lon 94 to 141
4. China (Tibet east, Sichuan, Yunnan, north China): lat 20 to 42, lon 100 to 128
5. Himalaya-India-Myanmar: lat 5 to 36, lon 75 to 100
6. West and Central Asia (Turkey, Iran, Afghanistan-Hindu Kush, Pakistan): lat 24 to 42, lon 26 to 75
Region centres (for the belt-observer crescent test) = centre of each box.

## Rules (7 per region; same definitions as before; Lahiri sidereal)
R1 Brihat Samhita circle (Sanskrit lists) by Moon nakshatra: chi-square vs background.
R2 Parashara Tantra ch.18 mandala by Moon nakshatra: chi-square.
R3 Moon within 5 deg of the Sun or a planet: binomial, upper tail.
R4 Moon in the sharp nakshatra group (Ardra, Aslesha, Jyeshtha, Mula): binomial upper tail.
R5 Sun in Vishakha: binomial upper tail.
R6 Level yoke crescent seen from the epicentre (BS 4.12 definition from preregistration_bs_crescent_true_horizon.md): binomial upper tail against random times at the same locations (3 per event).
R7 Same crescent condition seen from the region centre: date-shift null (500 shifts), upper tail.
Family = 6 regions x 7 rules = 42 tests per data set; Bonferroni bar 0.05/42 = 0.00119.

## Data sets
1990-2007 (discovery) and 2008-2026 (holdout) from the 9,368 declustered M>=5.5 events; Era A 1964-1989 (6,111); Era B 1900-1963 M>=7 (670). Sky rules use each period's matched six-hourly background.
## Decision rule
A region-rule pair is supported only if p < 0.00119 in 2008-2026 AND p < 0.05 in the same direction in 1964-1989. Descriptive otherwise. Power is limited by region size; each region's n is reported so a null can be read as "nothing found at this sensitivity".
