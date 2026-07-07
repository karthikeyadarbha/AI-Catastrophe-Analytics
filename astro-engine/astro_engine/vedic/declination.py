"""Declination phenomena -- the north/south dimension the zodiac ignores.

Ecliptic longitude says nothing about how far north or south a body swings.
This module exposes that axis, which is where the one (weak) lead of the tidal
study lived:

* **declination** of each body (already on the sample; surfaced here);
* **out-of-bounds** -- a body whose ``|declination|`` exceeds the obliquity of
  the ecliptic (~23.44 deg). The Moon can reach +-28.7 deg at a major standstill;
  OOB bodies are treated in some traditions as "unruly";
* **lunar nodal cycle** -- the 18.6-year regression of the Moon's node sets the
  envelope of lunar declination. Near a **major standstill** (ascending node
  near 0 deg Aries) the Moon's monthly declination swing is widest; near a
  **minor standstill** (node near Libra) it is narrowest. These are flagged.
"""
from __future__ import annotations

import numpy as np

from .sky import SkySample, STAR_PLANETS, wrap180
from .featureset import FeatureSet

_STANDSTILL_ORB = 20.0  # deg of node longitude around the equinoxes


def lunar_declination_envelope(obliquity_deg, node_trop_lon) -> np.ndarray:
    """Approximate max |lunar declination| for the current node position.

    Peaks at ``obliquity + 5.14 deg`` when the ascending node is at 0 deg Aries
    (major standstill) and dips to ``obliquity - 5.14 deg`` at 180 deg (minor).
    """
    return obliquity_deg + 5.145 * np.cos(np.radians(node_trop_lon))


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    obl = s.obliquity

    # Out-of-bounds flags for the Moon and the five star-planets.
    for body in ["Moon"] + STAR_PLANETS:
        fs.add_flag(f"{body}_out_of_bounds", np.abs(s.dec_deg[body]) > obl, "declination")

    # Northern vs southern hemisphere for the luminaries.
    fs.add_flag("moon_north_dec", s.dec_deg["Moon"] > 0, "declination")

    # Lunar nodal-cycle standstills (from the tropical ascending node = Rahu).
    node = s.trop_lon["Rahu"]
    near_major = np.minimum(node, 360.0 - node) <= _STANDSTILL_ORB
    dist_minor = np.abs(wrap180(node - 180.0))
    near_minor = dist_minor <= _STANDSTILL_ORB
    fs.add_flag("near_major_standstill", near_major, "declination")
    fs.add_flag("near_minor_standstill", near_minor, "declination")
    return fs
