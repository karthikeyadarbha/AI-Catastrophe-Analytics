"""Mundane cycles -- the slow outer-planet pair phases used in world astrology.

Mundane (world) astrology keys long-run events to the synodic cycles of the
slow planets -- above all the ~20-year Jupiter-Saturn conjunction, plus the
Saturn-Uranus, Saturn-Neptune, Saturn-Pluto, Uranus-Neptune, Uranus-Pluto and
Neptune-Pluto cycles. For each available pair this flags the hard and soft
phase points (conjunction / opposition / square / trine / sextile) of the faster
body relative to the slower, using tropical longitudes (phase is
ayanamsa-independent).

Requires the outer planets; pairs whose bodies are absent from the kernel are
skipped automatically.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .sky import SkySample, sep_deg
from .featureset import FeatureSet

#: Slow bodies, ordered slowest-first so pair names read (slow, fast)-agnostic.
_SLOW = ["Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
_PHASES = {"conjunction": (0.0, 6.0), "sextile": (60.0, 4.0), "square": (90.0, 5.0),
           "trine": (120.0, 5.0), "opposition": (180.0, 6.0)}


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    present = [b for b in _SLOW if b in s.trop_lon]
    for a, b in combinations(present, 2):
        sep = sep_deg(s.trop_lon[a], s.trop_lon[b])
        for name, (angle, orb) in _PHASES.items():
            hit = sep <= orb if angle == 0.0 else np.abs(sep - angle) <= orb
            fs.add_flag(f"cycle_{a}_{b}_{name}", hit, "mundane_cycle")
    return fs
