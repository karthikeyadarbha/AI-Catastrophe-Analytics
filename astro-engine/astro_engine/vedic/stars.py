"""Fixed stars -- conjunctions of luminaries/Lagna with prominent stars.

A curated set of bright, astrologically-weighted fixed stars (plus the Galactic
Centre) with their approximate **sidereal (Lahiri)** ecliptic longitudes near
epoch J2000. In a sidereal frame fixed stars barely move (proper motion is tiny
over a century), so a static table conjoined within a ~2 deg orb is adequate for
a screening study; it is **not** precision astrometry.

The anchor is Spica (Chitra) at 180.00 deg -- the star that defines the Lahiri
ayanamsa -- which also validates the table's zero point.
"""
from __future__ import annotations

import numpy as np

from .sky import SkySample, sep_deg
from .featureset import FeatureSet

#: Approximate sidereal (Lahiri, ~J2000) ecliptic longitudes, degrees.
STAR_SID_LON = {
    "Pleiades": 36.0,     # Krittika / Alcyone
    "Aldebaran": 46.0,    # Rohini -- royal star (Watcher of the East)
    "Rigel": 53.0,
    "Betelgeuse": 65.0,
    "Sirius": 80.0,
    "Regulus": 126.0,     # Magha -- royal star (Watcher of the North)
    "Spica": 180.0,       # Chitra -- Lahiri anchor
    "Arcturus": 184.0,    # Swati
    "Antares": 226.0,     # Jyeshtha -- royal star (Watcher of the West)
    "GalacticCenter": 243.0,
    "Vega": 261.5,        # Abhijit
    "Fomalhaut": 310.0,   # royal star (Watcher of the South)
}

_ORB = 2.0


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    bodies = {"Sun": s.sid_lon["Sun"], "Moon": s.sid_lon["Moon"]}
    if s.has_location:
        bodies["Lagna"] = s.asc_sid
    for star, lon in STAR_SID_LON.items():
        for bname, blon in bodies.items():
            fs.add_flag(f"conj_{bname}_{star}", sep_deg(blon, lon) < _ORB, "fixed_star")
    return fs
