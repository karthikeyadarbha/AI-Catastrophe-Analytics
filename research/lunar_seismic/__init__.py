"""Lunar / tidal earthquake-correlation study.

A research pipeline that tests whether earthquake occurrence is modulated by the
Moon's local tidal geometry (altitude / hour angle, declination, distance and
the Sun-Moon configuration) at each epicenter, plus an exploratory sidereal
"astrological" battery. Every test is run against a matched random-time null and
corrected for multiple comparisons.

Modules:
    catalog      USGS ComCat fetcher with on-disk caching.
    decluster    Gardner-Knopoff aftershock removal (independent mainshocks).
    geometry     Per-event lunar/solar tidal geometry via Skyfield.
    nulls        Matched random-time control samples.
    stats        Schuster test, permutation tests, Benjamini-Hochberg FDR.
"""

__all__ = ["catalog", "decluster", "geometry", "nulls", "stats"]
