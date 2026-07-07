"""Graha bala / avastha -- the dignity and state of each planet.

Combines several classical notions of planetary strength and condition:

* **dignity** -- the planet's relationship to the sign it sits in: exalted,
  own-sign, debilitated, or (via natural friendship with the sign's ruler)
  friendly / neutral / inimical;
* **combustion** -- too close to the Sun (astamgata); uses the same per-planet
  orb table as the scalar ``combustion`` plugin;
* **graha yuddha** -- a "planetary war" when two star-planets are within ~1 deg;
* **stationary** -- a planet near a retrograde station (|speed| ~ 0), classically
  a moment of concentrated strength.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np

from .sky import SkySample, STAR_PLANETS, sep_deg
from .featureset import FeatureSet
from . import tables as T

_DIGNITY_STATES = ["exalted", "own", "friend", "neutral", "enemy", "debilitated"]
_DIGNITY_PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
_STATIONARY_DEG_PER_DAY = 0.05  # ~3 arcmin/day; near a station
_YUDDHA_ORB = 1.0

_COMBUSTION_PATH = (Path(__file__).resolve().parent.parent / "config" / "combustion_limits.json")


def _load_combustion_limits() -> Dict[str, Dict[str, float]]:
    try:
        return json.loads(_COMBUSTION_PATH.read_text())
    except (OSError, ValueError):
        # Sensible classical defaults (degrees) if the config is unavailable.
        return {
            "Moon": {"direct": 12.0}, "Mars": {"direct": 17.0},
            "Mercury": {"direct": 14.0, "retrograde": 12.0},
            "Jupiter": {"direct": 11.0}, "Venus": {"direct": 10.0, "retrograde": 8.0},
            "Saturn": {"direct": 15.0},
        }


_COMBUSTION_LIMITS = _load_combustion_limits()


def dignity_state(planet: str, sign_idx: np.ndarray) -> np.ndarray:
    """Index (0-5) into :data:`_DIGNITY_STATES` for a planet across signs."""
    sign_idx = np.asarray(sign_idx, dtype=int)
    out = np.full(sign_idx.shape, 3, dtype=int)  # default neutral
    lords = np.array(T.SIGN_LORDS)
    friends, enemies = T.NATURAL_FRIENDS[planet], T.NATURAL_ENEMIES[planet]
    lord_here = lords[sign_idx]
    out[np.isin(lord_here, list(friends))] = 2  # friend
    out[np.isin(lord_here, list(enemies))] = 4  # enemy
    out[np.isin(sign_idx, T.OWN_SIGNS[planet])] = 1
    out[sign_idx == T.EXALTATION[planet]] = 0
    out[sign_idx == T.DEBILITATION[planet]] = 5
    return out


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    sun = s.sid_lon["Sun"]

    for p in _DIGNITY_PLANETS:
        sign_idx = np.floor(s.sid_lon[p] / 30.0).astype(int) % 12
        fs.add_categorical(f"dignity_{p}", dignity_state(p, sign_idx), 6,
                           _DIGNITY_STATES, "dignity")

    # Combustion (proximity to the Sun), honouring direct/retrograde orbs.
    for p, limits in _COMBUSTION_LIMITS.items():
        if p not in s.sid_lon:
            continue
        orb = np.where(s.retro.get(p, np.zeros(s.n, bool)) & ("retrograde" in limits),
                       limits.get("retrograde", limits["direct"]), limits["direct"])
        fs.add_flag(f"{p}_combust", sep_deg(s.sid_lon[p], sun) < orb, "dignity")

    # Stationary (near a retrograde station).
    for p in STAR_PLANETS:
        fs.add_flag(f"{p}_stationary", np.abs(s.speed[p]) < _STATIONARY_DEG_PER_DAY, "dignity")

    # Graha yuddha (planetary war): two star-planets within ~1 deg of longitude.
    for a, b in combinations(STAR_PLANETS, 2):
        fs.add_flag(f"yuddha_{a}_{b}", sep_deg(s.sid_lon[a], s.sid_lon[b]) < _YUDDHA_ORB, "dignity")
    return fs
