"""VedicFeatures -- the facade that ties every sub-library together.

Two entry points:

* :meth:`VedicFeatures.compute` -- vectorized feature extraction for an array of
  instants (the engine behind the earthquake battery). It builds one
  :class:`~astro_engine.vedic.sky.SkySample` and merges the
  :class:`~astro_engine.vedic.featureset.FeatureSet` of every enabled module.
* :meth:`VedicFeatures.chart` -- a single, human-readable natal/mundane chart for
  one instant and place (planets with sign/nakshatra/house/dignity/navamsa, the
  panchanga, the ascendant, Gulika, and the Vimshottari mahadasha timeline).

Selecting modules::

    vf = VedicFeatures()                          # everything
    vf = VedicFeatures(include=["panchanga", "declination"])
    vf = VedicFeatures(exclude=["upagraha", "stars"])
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

from .sky import SkySampler, SkySample, GRAHAS, STAR_PLANETS
from .featureset import FeatureSet
from . import (signs, panchanga, bhava, varga, dasha, dignity, aspects,
               declination, cycles, stars, upagraha, tables as T)

#: name -> module, in a natural reporting order.
MODULES = {
    "signs": signs, "panchanga": panchanga, "bhava": bhava, "varga": varga,
    "dasha": dasha, "dignity": dignity, "aspects": aspects,
    "declination": declination, "cycles": cycles, "stars": stars,
    "upagraha": upagraha,
}


def _default_kernel() -> str:
    return os.environ.get("ASTRO_KERNEL", "de421.bsp")


class VedicFeatures:
    def __init__(self, kernel_path: Optional[str] = None, *,
                 include: Optional[Sequence[str]] = None,
                 exclude: Optional[Sequence[str]] = None,
                 include_outer: bool = True):
        self.sampler = SkySampler(kernel_path or _default_kernel(),
                                  include_outer=include_outer)
        names = list(include) if include else list(MODULES)
        if exclude:
            names = [n for n in names if n not in set(exclude)]
        unknown = set(names) - set(MODULES)
        if unknown:
            raise ValueError(f"Unknown module(s): {sorted(unknown)}")
        self.modules = names

    # -- vectorized battery path -------------------------------------------
    def sample(self, times, latitude=None, longitude=None) -> SkySample:
        return self.sampler.sample(times, latitude, longitude)

    def compute(self, times, latitude=None, longitude=None) -> FeatureSet:
        """Merge every enabled module's features for ``times`` into one set."""
        s = self.sample(times, latitude, longitude)
        return self.compute_from_sample(s)

    def compute_from_sample(self, s: SkySample) -> FeatureSet:
        fs = FeatureSet()
        for name in self.modules:
            fs.merge(MODULES[name].features(s))
        return fs

    # -- single human-readable chart ---------------------------------------
    def chart(self, when: datetime, latitude: float, longitude: float,
              dasha_levels: int = 1) -> Dict:
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        s = self.sample([when], latitude, longitude)
        return _describe_chart(s, when, dasha_levels)


def _deg_in_sign(lon: float):
    return SIGN_TUPLE(int(lon // 30) % 12, lon % 30.0)


def SIGN_TUPLE(sign_idx, deg):
    return {"sign": T.SIGN_NAMES[sign_idx], "sign_index": sign_idx, "deg_in_sign": round(deg, 4)}


def _describe_chart(s: SkySample, when: datetime, dasha_levels: int) -> Dict:
    asc_sign = int(s.asc_sid[0] // 30) % 12
    planets: Dict[str, Dict] = {}
    for g in GRAHAS:
        lon = float(s.sid_lon[g][0])
        sign_idx = int(lon // 30) % 12
        nak = int((lon % 360) // (360 / 27)) % 27
        pada = int((lon % (360 / 27)) // (360 / 108)) % 4
        entry = {
            **SIGN_TUPLE(sign_idx, lon % 30.0),
            "longitude": round(lon, 4),
            "nakshatra": T.NAKSHATRA_NAMES[nak],
            "pada": pada + 1,
            "retrograde": bool(s.retro[g][0]),
            "house": (sign_idx - asc_sign) % 12 + 1,
            "navamsa": T.SIGN_NAMES[int(varga.varga_sign(np.array([lon]), 9)[0])],
            "declination": round(float(s.dec_deg[g][0]), 4),
            "speed": round(float(s.speed[g][0]), 5),
        }
        if g in dignity._DIGNITY_PLANETS:
            state = dignity.dignity_state(g, np.array([sign_idx]))[0]
            entry["dignity"] = dignity._DIGNITY_STATES[int(state)]
        planets[g] = entry

    sun, moon = s.sid_lon["Sun"][0], s.sid_lon["Moon"][0]
    tithi_i = int(panchanga.tithi_index(sun, moon))
    pan = {
        "tithi": f"{'Krishna' if tithi_i >= 15 else 'Shukla'} {T.TITHI_NAMES[tithi_i % 15]}",
        "vara": T.VARA_NAMES[int(panchanga.vara_index(s.local_dow)[0])],
        "yoga": T.YOGA_NAMES[int(panchanga.yoga_index(sun, moon))],
        "karana": T.KARANA_NAMES[int(T.karana_name_index(panchanga.half_tithi_index(sun, moon))[0])],
        "nakshatra": T.NAKSHATRA_NAMES[int(panchanga.nakshatra_index(moon))],
    }

    try:
        gulika_lon = float(upagraha.gulika_longitude(s)[0])
        gsid = (gulika_lon - float(s.ayanamsa[0])) % 360.0
        gulika = SIGN_TUPLE(int(gsid // 30) % 12, gsid % 30.0)
    except Exception:
        gulika = None

    maha = dasha.vimshottari_periods(float(moon), when, levels=dasha_levels)
    dasha_out = [
        {"lord": m[0], **({"sub": m[1]} if dasha_levels > 1 else {}),
         "start": m[-2].isoformat(), "end": m[-1].isoformat()}
        for m in maha
    ]

    return {
        "datetime_utc": when.astimezone(timezone.utc).isoformat(),
        "location": {"latitude": float(s.latitude[0]), "longitude": float(s.longitude[0])},
        "ayanamsa": round(float(s.ayanamsa[0]), 6),
        "ascendant": {**SIGN_TUPLE(asc_sign, float(s.asc_sid[0]) % 30.0),
                      "longitude": round(float(s.asc_sid[0]), 4)},
        "planets": planets,
        "panchanga": pan,
        "gulika": gulika,
        "vimshottari": dasha_out,
    }
