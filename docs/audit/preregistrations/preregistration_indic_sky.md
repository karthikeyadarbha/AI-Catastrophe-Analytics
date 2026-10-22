# Pre-registration: re-running both experiments on the Indian-method sky (indic-ganita-engine v1.4.2)
Written 22 October 2026, after the engine was frozen (v1.4.2, combined hash 002454975b686ef4) and before any experiment was run on it.

## Question
Every test in the Rule-First and Data-First studies used a modern sky (Swiss Ephemeris via the astral tool, or `shared/myeph.py` with Lahiri ayanamsa). The texts computed their sky differently. Does any verdict change when the sky is computed the way the Kerala and Siddhantic texts computed it?

## What changes, and what does not
* Only the sky source changes. `shared/myeph.py` gets a switch: `SKY_ENGINE=modern` (default, unchanged) or `ts`, `drk`, `ss`. Under the switch, every sidereal longitude (7 planets and Rahu), retrograde flag and planet speed comes from the engine, in the engine's own sidereal frame (the frame implied by the dhruvas). Tropical values are formed as engine sidereal + the same Lahiri ayanamsa, so a script that subtracts the ayanamsa gets the engine's sidereal longitude back.
* Scripts, rules, nakshatra lists, catalogues, background grids, seeds and bars are not changed. Eclipses stay physical (`shared/eclipses.py` is not switched): an eclipse is an observed event, and the engine agrees with the modern times within 0.5 h (FREEZE.md).
* Primary engine: TS (Tantrasangraha parameters, Madhava Rsine). Cross-checks: Drk (Paramesvara 1431) and SS (Suryasiddhanta). For 1850-2025 SS has known offsets (Jupiter +4.5 deg, Saturn -5.1 deg, Venus rms 6 deg); its results are reported as a sensitivity check only.

## Scope
All scripts of both experiments that read the sky through `shared/myeph.py` and run from `shared/inputs/`. Scripts that read sky values already computed by the astral tool (`ch14_16.py` with its stored Deadly 15 longitudes; the Modern M6.5+ scripts needing `quakes_m65_sid.csv` and `bg_sid.csv` features) cannot be switched without the astral tool and are listed as not covered.

## Decision rule
For each test, the original verdict (pass or fail at its own pre-registered bar, holdout and direction conditions included) is compared with the verdict under TS. The result of this study is the list of tests whose verdict changes. A change under TS that is not repeated under Drk is reported as engine-dependent. No new test is added and no bar is adjusted. If no verdict changes, the conclusion of both studies holds on the Indian-method sky.

## Also reported
For each script, the fraction of events whose Moon nakshatra (and, where used, planet nakshatra or sign) differs between the modern and the TS sky, since that is the only route by which a verdict can change.

## Addition before any run (22 October 2026)
Crescent sighting (`shared/crescent.py`, `shared/moonlat.py`) is an observed event like an eclipse, so it also stays on the modern sky under every `SKY_ENGINE` setting.
