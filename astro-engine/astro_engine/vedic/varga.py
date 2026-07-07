"""Varga (divisional charts) -- the sign a graha occupies in the sub-charts.

A varga ``Dn`` subdivides each 30 deg sign into ``n`` parts and re-maps each part
to a sign by a scheme-specific rule. The Rasi (D1) and Navamsa (D9) are the two
most weighted charts in Vedic astrology; several others are widely used.
Implemented here: D1, D2 (Hora), D3 (Drekkana), D9 (Navamsa), D10 (Dasamsa),
D12 (Dwadasamsa), D30 (Trimsamsa), D60 (Shashtiamsa).

Each function maps a **sidereal** longitude (array) to a sign index 0-11.
"""
from __future__ import annotations

import numpy as np

from .sky import SkySample, GRAHAS
from .featureset import FeatureSet
from . import tables as T

_SUPPORTED = (1, 2, 3, 9, 10, 12, 30, 60)


def _base_and_offset(lon):
    lon = np.asarray(lon, dtype=float) % 360.0
    base = np.floor(lon / 30.0).astype(int) % 12   # rasi sign
    within = lon % 30.0                             # 0..30 within the sign
    return lon, base, within


def varga_sign(lon, n: int) -> np.ndarray:
    """Sign index (0-11) of the ``Dn`` varga for a sidereal longitude array."""
    lon, base, within = _base_and_offset(lon)
    odd = (base % 2) == 0  # Aries(0) is "odd" in 1-based counting

    if n == 1:
        return base
    if n == 2:  # Hora: Sun's (Leo=4) / Moon's (Cancer=3) hora
        first_half = within < 15.0
        # odd sign: 1st half -> Leo, 2nd -> Cancer; even sign: reversed
        leo = np.where(odd, first_half, ~first_half)
        return np.where(leo, 4, 3)
    if n == 3:  # Drekkana: 1st same, 2nd 5th (+4), 3rd 9th (+8)
        part = np.floor(within / 10.0).astype(int)
        return (base + part * 4) % 12
    if n == 9:  # Navamsa: continuous 3deg20' parts around the zodiac
        return np.floor(lon / (10.0 / 3.0)).astype(int) % 12
    if n == 10:  # Dasamsa: odd from same sign, even from the 9th (+8)
        part = np.floor(within / 3.0).astype(int)
        start = np.where(odd, base, (base + 8) % 12)
        return (start + part) % 12
    if n == 12:  # Dwadasamsa: from the same sign
        part = np.floor(within / 2.5).astype(int)
        return (base + part) % 12
    if n == 30:  # Trimsamsa: unequal 5/5/8/7/5 rulership split
        return _trimsamsa(base, within, odd)
    if n == 60:  # Shashtiamsa: 0.5deg parts counted from the sign
        part = np.floor(within / 0.5).astype(int)
        return (base + part) % 12
    raise ValueError(f"Unsupported varga D{n}; supported: {_SUPPORTED}")


def _trimsamsa(base, within, odd) -> np.ndarray:
    # Odd signs: Mars(Ar) Sat(Aq) Jup(Sg) Merc(Ge) Ven(Li) over 5/5/8/7/5 deg.
    odd_bounds = np.array([5.0, 10.0, 18.0, 25.0, 30.0])
    odd_signs = np.array([0, 10, 8, 2, 6])
    # Even signs: reversed rulers -> Ven(Ta) Merc(Vi) Jup(Pi) Sat(Cp) Mars(Sc).
    even_bounds = np.array([5.0, 12.0, 20.0, 25.0, 30.0])
    even_signs = np.array([1, 5, 11, 9, 7])
    out = np.empty(within.shape, dtype=int)
    oi = np.searchsorted(odd_bounds, within, side="right").clip(max=4)
    ei = np.searchsorted(even_bounds, within, side="right").clip(max=4)
    out[odd] = odd_signs[oi[odd]]
    out[~odd] = even_signs[ei[~odd]]
    return out


def chart(lon, n: int) -> np.ndarray:
    """Alias for :func:`varga_sign` (reads naturally in chart code)."""
    return varga_sign(lon, n)


def features(s: SkySample) -> FeatureSet:
    """Battery contribution: the Navamsa (D9) sign of every graha."""
    fs = FeatureSet()
    for g in GRAHAS:
        fs.add_categorical(f"navamsa_{g}", varga_sign(s.sid_lon[g], 9), 12,
                           T.SIGN_NAMES, "navamsa")
    return fs
