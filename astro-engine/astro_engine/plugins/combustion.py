"""Detect combustion periods (a planet too close to the Sun)."""
from datetime import timedelta
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from astro_engine.core.interface import PluginInterface
from astro_engine.adapters.base import EphemerisEngineBase
from astro_engine.models.context import Context
from astro_engine.models.location import Location
from astro_engine.models.date import Date
from astro_engine.models.date_range import DateRange
from astro_engine.models.planet import PlanetName
from astro_engine.models.events import Period
from astro_engine.models.event_repository import EventRepository
from astro_engine.utils.config_loader import load_json

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "combustion_limits.json"

# Planets for which combustion is undefined: the Sun itself and the nodes.
_SKIP = (PlanetName.SUN, PlanetName.RAHU, PlanetName.KETU)


class CombustionPlugin(PluginInterface):
    """Detect combustion via a robust bracket-and-refine algorithm."""

    def __init__(self):
        limits = load_json(_CONFIG_PATH)
        self.limits: Dict[str, Any] = limits if limits is not None else {}

    @property
    def name(self) -> str:
        return "combustion"

    def run(self, context: Context, ephemeris: EphemerisEngineBase) -> EventRepository:
        master_repo = EventRepository()
        planets_to_check = context.planets or list(PlanetName)

        for planet in planets_to_check:
            if planet in _SKIP:
                continue
            planet_repo = self._compute_for_single_planet(planet, context, ephemeris)
            master_repo.extend(planet_repo)
        return master_repo

    # --- Helper methods -------------------------------------------------- #

    def _get_angular_separation(
        self, planet: PlanetName, date: Date, ephemeris: EphemerisEngineBase, location: Location
    ) -> float:
        """Shortest angular distance (degrees) between the planet and the Sun."""
        planet_lon = ephemeris.get_planet_longitude(planet, date, location)
        sun_lon = ephemeris.get_planet_longitude(PlanetName.SUN, date, location)
        diff = abs(planet_lon - sun_lon)
        return min(diff, 360 - diff)

    def _get_limit_for_planet(
        self, planet: PlanetName, date: Date, ephemeris: EphemerisEngineBase, location: Location
    ) -> float:
        """Combustion limit for the planet, honouring direct/retrograde values."""
        planet_limits = self.limits.get(planet.value)
        if not planet_limits:
            raise ValueError(f"Combustion limits not defined for {planet.value}")

        if "retrograde" not in planet_limits:
            return planet_limits["direct"]

        motion = ephemeris.get_planet_motion(planet, date, location)
        return planet_limits.get(motion.value, planet_limits["direct"])

    def _is_combust(
        self, planet: PlanetName, date: Date, ephemeris: EphemerisEngineBase, location: Location
    ) -> bool:
        separation = self._get_angular_separation(planet, date, ephemeris, location)
        limit = self._get_limit_for_planet(planet, date, ephemeris, location)
        return separation < limit

    def _find_bracket(
        self, planet: PlanetName, anchor_date: Date, forward: bool,
        ephemeris: EphemerisEngineBase, location: Location,
    ) -> Optional[Tuple[Date, Date]]:
        """Walk until the combustion state flips; return a bracketing interval."""
        step = timedelta(days=1) if forward else timedelta(days=-1)
        initial_state = self._is_combust(planet, anchor_date, ephemeris, location)

        prev_date = anchor_date
        for _ in range(730):  # up to ~2 years covers any planet's cycle
            curr_date = prev_date + step
            current_state = self._is_combust(planet, curr_date, ephemeris, location)
            if current_state != initial_state:
                return (prev_date, curr_date) if forward else (curr_date, prev_date)
            prev_date = curr_date
        return None

    def _refine(
        self, planet: PlanetName, d1: Date, d2: Date,
        ephemeris: EphemerisEngineBase, location: Location, tol: float = 1e-6,
    ) -> Date:
        """Bisection for the instant the planet crosses the combustion boundary."""
        is_combust_at_d2 = self._is_combust(planet, d2, ephemeris, location)

        for _ in range(60):
            mid = Date(d1.to_datetime() + (d2.to_datetime() - d1.to_datetime()) / 2)
            separation = self._get_angular_separation(planet, mid, ephemeris, location)
            limit = self._get_limit_for_planet(planet, mid, ephemeris, location)

            if abs(separation - limit) < tol:
                return mid

            if self._is_combust(planet, mid, ephemeris, location) == is_combust_at_d2:
                d2 = mid
            else:
                d1 = mid

        return Date(d1.to_datetime() + (d2.to_datetime() - d1.to_datetime()) / 2)

    # --- Main computation ------------------------------------------------ #

    def _compute_for_single_planet(
        self, planet: PlanetName, context: Context, ephemeris: EphemerisEngineBase
    ) -> EventRepository:
        repo = EventRepository()
        start, end, location = context.date_range.start, context.date_range.end, context.location

        is_combust_at_start = self._is_combust(planet, start, ephemeris, location)
        combustion_start_date: Optional[Date] = None

        if is_combust_at_start:
            bracket = self._find_bracket(planet, start, forward=False, ephemeris=ephemeris, location=location)
            combustion_start_date = (
                self._refine(planet, bracket[0], bracket[1], ephemeris, location) if bracket else start
            )

        prev_date = start
        was_combust = is_combust_at_start

        curr_date = start + timedelta(days=1)
        while curr_date <= end:
            is_combust = self._is_combust(planet, curr_date, ephemeris, location)

            if not was_combust and is_combust:
                combustion_start_date = self._refine(planet, prev_date, curr_date, ephemeris, location)

            elif was_combust and not is_combust:
                if combustion_start_date:
                    combustion_end_date = self._refine(planet, prev_date, curr_date, ephemeris, location)
                    repo.add(Period(
                        event_type=self.name,
                        planet=planet,
                        date_range=DateRange(combustion_start_date, combustion_end_date),
                    ))
                combustion_start_date = None

            prev_date = curr_date
            was_combust = is_combust
            curr_date += timedelta(days=1)

        if was_combust and combustion_start_date:
            bracket = self._find_bracket(planet, end, forward=True, ephemeris=ephemeris, location=location)
            combustion_end_date = (
                self._refine(planet, bracket[0], bracket[1], ephemeris, location) if bracket else end
            )
            repo.add(Period(
                event_type=self.name,
                planet=planet,
                date_range=DateRange(combustion_start_date, combustion_end_date),
            ))

        return repo
