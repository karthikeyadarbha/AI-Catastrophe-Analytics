"""Backend sanity checks and Swiss-vs-JPL parity."""
import pytest

from astro_engine.models.planet import PlanetName
from astro_engine.models.zodiac_sign import ZodiacSign
from astro_engine.models.motion import MotionType

from .conftest import requires_swiss, requires_skyfield

LOC = (17.385, 78.4867)  # Hyderabad
WHEN = "2025-01-01T05:30:00+05:30"

# Approximate mean daily motions (deg/day) used only for loose sanity bounds.
_SPEED_BOUNDS = {
    PlanetName.SUN: (0.9, 1.1),
    PlanetName.MOON: (11.0, 15.5),
}


def _check_position(pos):
    assert 0.0 <= pos.longitude < 360.0
    assert isinstance(pos.rasi, ZodiacSign)
    assert 1 <= pos.pada <= 4
    assert pos.dms  # non-empty formatted string


@requires_swiss
@pytest.mark.swiss
class TestSwissBackend:
    def test_all_planets_have_valid_positions(self, swiss_engine):
        for p in PlanetName:
            _check_position(swiss_engine.position(p, WHEN, LOC))

    def test_sun_and_moon_speeds(self, swiss_engine):
        for planet, (lo, hi) in _SPEED_BOUNDS.items():
            speed = swiss_engine.position(planet, WHEN, LOC).speed
            assert lo <= speed <= hi

    def test_nodes_are_retrograde(self, swiss_engine):
        # Mean lunar nodes always move retrograde.
        for node in (PlanetName.RAHU, PlanetName.KETU):
            assert swiss_engine.ephemeris.get_planet_motion(
                node, _date(node, swiss_engine), _loc()
            ) == MotionType.RETROGRADE

    def test_ketu_opposite_rahu(self, swiss_engine):
        rahu = swiss_engine.longitude("Rahu", WHEN, LOC)
        ketu = swiss_engine.longitude("Ketu", WHEN, LOC)
        diff = (ketu - rahu) % 360.0
        assert abs(diff - 180.0) < 1e-6


@requires_skyfield
@pytest.mark.jpl
class TestJplBackend:
    def test_all_planets_have_valid_positions(self, jpl_engine):
        for p in PlanetName:
            _check_position(jpl_engine.position(p, WHEN, LOC))

    def test_ketu_opposite_rahu(self, jpl_engine):
        rahu = jpl_engine.longitude("Rahu", WHEN, LOC)
        ketu = jpl_engine.longitude("Ketu", WHEN, LOC)
        diff = (ketu - rahu) % 360.0
        assert abs(diff - 180.0) < 1e-6


@requires_swiss
@requires_skyfield
@pytest.mark.jpl
@pytest.mark.swiss
class TestParity:
    """JPL (Skyfield + Lahiri) must match Swiss Ephemeris closely."""

    def test_longitude_parity(self, swiss_engine, jpl_engine):
        max_arcsec = 0.0
        for p in PlanetName:
            a = swiss_engine.longitude(p, WHEN, LOC)
            b = jpl_engine.longitude(p, WHEN, LOC)
            d = abs(((a - b + 180.0) % 360.0) - 180.0) * 3600.0
            max_arcsec = max(max_arcsec, d)
        # Sub-arcsecond for planets; the Moon (fastest, parallax-sensitive) is
        # the worst case at ~1". Allow a comfortable 10" margin.
        assert max_arcsec < 10.0, f"max longitude divergence {max_arcsec:.2f}\""

    def test_motion_agreement(self, swiss_engine, jpl_engine):
        # Mercury is retrograde on 2025-03-20; both backends must agree.
        when = "2025-03-20T12:00:00+05:30"
        from astro_engine.models.date import Date
        from astro_engine.models.location import Location

        date = Date.from_iso(when)
        loc = Location(LOC[0], LOC[1])
        for p in (PlanetName.MERCURY, PlanetName.SUN, PlanetName.SATURN):
            m_sw = swiss_engine.ephemeris.get_planet_motion(p, date, loc)
            m_jp = jpl_engine.ephemeris.get_planet_motion(p, date, loc)
            assert m_sw == m_jp, f"{p.value}: swiss={m_sw} jpl={m_jp}"


# --- small helpers for the swiss node test (need Date/Location objects) ------
def _date(_planet, _engine):
    from astro_engine.models.date import Date

    return Date.from_iso(WHEN)


def _loc():
    from astro_engine.models.location import Location

    return Location(LOC[0], LOC[1])
