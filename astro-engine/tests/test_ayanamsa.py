"""Validate the independent Lahiri ayanamsa against Swiss Ephemeris."""
import pytest

from astro_engine.utils.ayanamsa import (
    get_ayanamsa,
    julian_day_ut,
    lahiri_ayanamsa,
)
from datetime import datetime, timezone

from .conftest import requires_swiss


def test_julian_day_ut_j2000():
    # 2000-01-01T12:00:00Z is exactly JD 2451545.0 (J2000.0).
    jd = julian_day_ut(datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    assert abs(jd - 2451545.0) < 1e-6


def test_get_ayanamsa_unknown_mode_raises():
    with pytest.raises(NotImplementedError):
        get_ayanamsa("Raman", 2451545.0)


def test_lahiri_at_j2000_is_reasonable():
    # Lahiri ayanamsa at J2000 is ~23.85 deg.
    assert 23.8 < lahiri_ayanamsa(2451545.0) < 23.9


@requires_swiss
@pytest.mark.swiss
def test_lahiri_matches_swisseph():
    """The fitted cubic must reproduce swe SIDM_LAHIRI to < 0.001 arcsec."""
    import swisseph as swe

    swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)

    max_err_arcsec = 0.0
    for year in range(1900, 2101, 5):
        dt = datetime(year, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        jd = julian_day_ut(dt)
        ours = lahiri_ayanamsa(jd)
        theirs = swe.get_ayanamsa_ut(jd)
        err = abs(ours - theirs) * 3600.0
        max_err_arcsec = max(max_err_arcsec, err)

    assert max_err_arcsec < 0.001, f"max deviation {max_err_arcsec:.6f}\" exceeds 0.001\""
