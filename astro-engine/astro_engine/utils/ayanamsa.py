"""Independent Lahiri ayanamsa (precession of the equinoxes), swisseph-free.

The sidereal zodiac used in Vedic astrology is offset from the tropical
(equinox-based) zodiac by the *ayanamsa*, which grows with the precession of
the equinoxes (~50.3 arcsec/year). This module provides the **Lahiri**
(Chitrapaksha) ayanamsa as a cubic polynomial in Julian centuries from J2000.

The coefficients were obtained by a least-squares fit to the Swiss Ephemeris
``swe.get_ayanamsa_ut`` (SIDM_LAHIRI) sampled quarterly from 1800-2100. The fit
reproduces Swiss Ephemeris to better than **0.0001 arcsec** across that range,
so the JPL backend needs no dependency on ``pyswisseph`` to produce Lahiri
sidereal longitudes that agree with the Swiss backend. The leading term
(a1 = 1.39689 deg/century = 5028.80"/century) is exactly the IAU general
precession rate, confirming the expansion is physically meaningful rather than
an arbitrary curve fit.

See ``tests/test_ayanamsa.py`` for the validation against Swiss Ephemeris.
"""
from datetime import datetime, timezone
from typing import Callable, Dict

# Unix epoch (1970-01-01T00:00:00Z) expressed as a Julian Day number.
_JD_UNIX_EPOCH = 2440587.5
_JD_J2000 = 2451545.0
_DAYS_PER_CENTURY = 36525.0

# ayanamsa(deg) = A0 + A1*T + A2*T^2 + A3*T^3, T = Julian centuries from J2000.
# Fit to Swiss Ephemeris SIDM_LAHIRI, max deviation < 0.0001" over 1800-2100.
_LAHIRI_COEFFS = (
    23.857092350952282,      # A0  (ayanamsa at J2000.0)
    1.3968879522407693,      # A1  (= 5028.80"/century, general precession)
    3.0709416229379727e-04,  # A2
    9.690116264948756e-09,   # A3
)


def julian_day_ut(dt: datetime) -> float:
    """Return the UT (UTC) Julian Day number for a datetime.

    Naive datetimes are assumed to be UTC. UTC and UT1 differ by < 0.9 s, which
    is negligible for the ayanamsa (< 1e-6 arcsec).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _JD_UNIX_EPOCH + dt.timestamp() / 86400.0


def lahiri_ayanamsa(jd_ut: float) -> float:
    """Lahiri ayanamsa in degrees for a UT Julian Day number."""
    t = (jd_ut - _JD_J2000) / _DAYS_PER_CENTURY
    a0, a1, a2, a3 = _LAHIRI_COEFFS
    return a0 + a1 * t + a2 * t * t + a3 * t * t * t


def lahiri_ayanamsa_at(dt: datetime) -> float:
    """Lahiri ayanamsa in degrees for a datetime."""
    return lahiri_ayanamsa(julian_day_ut(dt))


# Registry of supported ayanamsa modes for the pure-Python (JPL) backend.
AYANAMSA_FUNCS: Dict[str, Callable[[float], float]] = {
    "Lahiri": lahiri_ayanamsa,
    "default": lahiri_ayanamsa,
}


def get_ayanamsa(mode: str, jd_ut: float) -> float:
    """Return the ayanamsa (degrees) for a named mode and UT Julian Day.

    Raises:
        NotImplementedError: If ``mode`` is not implemented by the pure-Python
            backend. Use the Swiss backend for other ayanamsas.
    """
    func = AYANAMSA_FUNCS.get(mode)
    if func is None:
        raise NotImplementedError(
            f"Ayanamsa mode '{mode}' is not implemented by the pure-Python "
            f"(JPL) backend. Supported modes: {sorted(AYANAMSA_FUNCS)}. "
            f"Use backend='swiss' for other ayanamsas."
        )
    return func(jd_ut)
