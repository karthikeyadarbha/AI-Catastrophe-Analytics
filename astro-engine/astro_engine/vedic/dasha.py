"""Vimshottari dasha -- the 120-year planetary period system.

The Moon's nakshatra at a moment fixes the running **mahadasha** (major-period)
lord, and the Moon's fractional progress through that nakshatra fixes how much
of the period is already elapsed. This module provides:

* :func:`nakshatra_lord_index` / the ``moon_dasha_lord`` battery feature -- the
  graha "ruling" the moment (the mahadasha lord if the moment were a birth);
* :func:`vimshottari_periods` -- the full mahadasha (and optional antardasha)
  timeline for a chart, given the Moon's longitude and a birth instant.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

from .sky import SkySample
from .featureset import FeatureSet
from . import tables as T

_NAK_WIDTH = 360.0 / 27.0
_SIDEREAL_YEAR_DAYS = 365.25  # dasha years are conventionally 365.25 days


def nakshatra_lord_index(moon_sid_lon) -> np.ndarray:
    """Index (0-8) into :data:`NAKSHATRA_LORD_ORDER` of the ruling dasha lord."""
    nak = np.floor(np.asarray(moon_sid_lon) / _NAK_WIDTH).astype(int) % 27
    return nak % 9


def vimshottari_periods(moon_sid_lon: float, birth: datetime,
                        levels: int = 1) -> List[Tuple]:
    """Return the mahadasha timeline as ``(lord, start, end)`` tuples.

    Args:
        moon_sid_lon: Moon's sidereal longitude at birth (degrees).
        birth: Birth instant (tz-aware recommended).
        levels: ``1`` for mahadashas only; ``2`` to also nest antardashas as
            ``(maha_lord, antar_lord, start, end)``.
    """
    order = T.NAKSHATRA_LORD_ORDER
    nak = int(moon_sid_lon // _NAK_WIDTH) % 27
    start_lord = nak % 9
    frac_elapsed = (moon_sid_lon % _NAK_WIDTH) / _NAK_WIDTH

    seq = [order[(start_lord + i) % 9] for i in range(9)]
    first_years = T.VIMSHOTTARI_YEARS[seq[0]]
    cursor = birth - timedelta(days=first_years * frac_elapsed * _SIDEREAL_YEAR_DAYS)

    out: List[Tuple] = []
    for lord in seq:
        span = timedelta(days=T.VIMSHOTTARI_YEARS[lord] * _SIDEREAL_YEAR_DAYS)
        maha_start, maha_end = cursor, cursor + span
        if levels <= 1:
            out.append((lord, maha_start, maha_end))
        else:
            sub_cursor = maha_start
            for j in range(9):
                sub = order[(order.index(lord) + j) % 9]
                sub_years = (T.VIMSHOTTARI_YEARS[lord] * T.VIMSHOTTARI_YEARS[sub]
                             / T.VIMSHOTTARI_TOTAL)
                sub_span = timedelta(days=sub_years * _SIDEREAL_YEAR_DAYS)
                out.append((lord, sub, sub_cursor, sub_cursor + sub_span))
                sub_cursor += sub_span
        cursor = maha_end
    return out


def features(s: SkySample) -> FeatureSet:
    fs = FeatureSet()
    lord = nakshatra_lord_index(s.sid_lon["Moon"])
    fs.add_categorical("moon_dasha_lord", lord, 9, T.NAKSHATRA_LORD_ORDER, "dasha")
    return fs
