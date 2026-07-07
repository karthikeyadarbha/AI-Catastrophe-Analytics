"""Aspects -- angular relationships, in three complementary systems.

1. **Western / Ptolemaic** (degree-based, symmetric): conjunction 0 deg,
   sextile 60, square 90, trine 120, opposition 180, each within an orb. Applied
   to every graha pair.
2. **Vedic graha drishti** (sign-based, asymmetric): every graha aspects the 7th
   sign from itself; Mars also the 4th & 8th, Jupiter the 5th & 9th, Saturn the
   3rd & 10th (nodes treated like Jupiter). We surface the *special* aspects
   (those unique to the Vedic system) that fall on the luminaries or the Lagna.
3. **Declination aspects**: a **parallel** (same declination) or
   **contraparallel** (equal and opposite) within a tight orb -- the north/south
   analogue of a conjunction/opposition, centred here on the Sun and Moon.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .sky import SkySample, GRAHAS, STAR_PLANETS, sep_deg
from .featureset import FeatureSet

#: Western aspect angles and their orbs (degrees).
PTOLEMAIC = {"conjunction": (0.0, 8.0), "sextile": (60.0, 4.0), "square": (90.0, 6.0),
             "trine": (120.0, 6.0), "opposition": (180.0, 8.0)}

#: Special Vedic full-aspect houses (beyond the universal 7th) per graha.
SPECIAL_DRISHTI = {"Mars": (4, 8), "Jupiter": (5, 9), "Saturn": (3, 10),
                   "Rahu": (5, 9), "Ketu": (5, 9)}

_DEC_PARALLEL_ORB = 1.0


def _aspect_pairs():
    pairs = combinations(GRAHAS, 2)
    return [(a, b) for a, b in pairs if {a, b} != {"Rahu", "Ketu"}]  # drop degenerate 180


def _western(fs: FeatureSet, s: SkySample) -> None:
    for a, b in _aspect_pairs():
        sep = sep_deg(s.sid_lon[a], s.sid_lon[b])
        for name, (angle, orb) in PTOLEMAIC.items():
            if angle == 180.0 and {a, b} == {"Rahu", "Ketu"}:
                continue
            hit = sep <= orb if angle == 0.0 else np.abs(sep - angle) <= orb
            fs.add_flag(f"{name}_{a}_{b}", hit, "aspect_western")


def _drishti(fs: FeatureSet, s: SkySample) -> None:
    signs = {g: np.floor(s.sid_lon[g] / 30.0).astype(int) % 12 for g in GRAHAS}
    targets = {"Sun": signs["Sun"], "Moon": signs["Moon"]}
    if s.has_location:
        targets["Lagna"] = np.floor(s.asc_sid / 30.0).astype(int) % 12
    for aspecting, houses in SPECIAL_DRISHTI.items():
        for tname, tsign in targets.items():
            if aspecting == tname:
                continue
            # house counted 1..12 from the aspecting graha to the target sign
            house = (tsign - signs[aspecting]) % 12 + 1
            hit = np.isin(house, houses)
            fs.add_flag(f"drishti_{aspecting}_{tname}", hit, "aspect_vedic")


def _declination(fs: FeatureSet, s: SkySample) -> None:
    focus = ["Sun", "Moon"]
    others = STAR_PLANETS + ["Rahu"]
    seen = set()
    for f in focus:
        for o in focus + others:
            if o == f or (o, f) in seen or (f, o) in seen:
                continue
            seen.add((f, o))
            da, db = s.dec_deg[f], s.dec_deg[o]
            fs.add_flag(f"parallel_{f}_{o}", np.abs(da - db) < _DEC_PARALLEL_ORB, "aspect_declination")
            fs.add_flag(f"contraparallel_{f}_{o}", np.abs(da + db) < _DEC_PARALLEL_ORB, "aspect_declination")


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    _western(fs, s)
    _drishti(fs, s)
    _declination(fs, s)
    return fs
