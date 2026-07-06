"""Facade API and plugin event-detection smoke tests (Swiss backend)."""
import pytest

from astro_engine import AstroEngine
from astro_engine.models.planet import PlanetName
from astro_engine.models.planetary_position import PlanetaryPosition

from .conftest import requires_swiss

LOC = (17.385, 78.4867)


def test_available_backends():
    backends = AstroEngine.available_backends()
    assert isinstance(backends, list)
    assert "swiss" in backends or "jpl" in backends


@requires_swiss
@pytest.mark.swiss
class TestFacade:
    def test_plugins_listed(self, swiss_engine):
        for name in ("retrograde", "combustion", "rasi_transit", "nakshatra_transit"):
            assert name in swiss_engine.plugins

    def test_position_coercion(self, swiss_engine):
        # str date, tuple location, str planet all coerced.
        pos = swiss_engine.position("Mars", "2025-01-01", LOC)
        assert isinstance(pos, PlanetaryPosition)
        assert pos.planet == PlanetName.MARS

    def test_positions_defaults_to_nine(self, swiss_engine):
        allp = swiss_engine.positions("2025-01-01", LOC)
        assert len(allp) == len(list(PlanetName)) == 9

    def test_retrograde_period_detected(self, swiss_engine):
        # Mercury retrograde spanned 2025-03-15 -> 2025-04-07.
        evs = swiss_engine.find_events(
            start="2025-03-01", end="2025-04-30", location=LOC,
            plugins=["retrograde"], planets=["Mercury"],
        )
        periods = list(evs)
        assert len(periods) >= 1

    def test_sun_makara_transit(self, swiss_engine):
        # The Sun enters Makara (Capricorn) ~Jan 14 (Makara Sankranti).
        evs = swiss_engine.find_events(
            start="2025-01-01", end="2025-01-31", location=LOC,
            plugins=["rasi_transit"], planets=["Sun"],
        )
        transits = list(evs)
        assert len(transits) == 1
        assert transits[0].date.dt.day == 14
        assert transits[0].date.dt.month == 1

    def test_all_plugins_run_without_error(self, swiss_engine):
        evs = swiss_engine.find_events(
            start="2025-01-01", end="2025-01-05", location=LOC,
            planets=["Moon", "Mercury"],
        )
        # Should at least detect Moon nakshatra/pada ingresses in 5 days.
        assert len(list(evs)) > 0
