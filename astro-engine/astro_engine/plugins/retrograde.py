"""Detect retrograde periods for planets."""
from datetime import timedelta
from typing import Optional, Tuple

from astro_engine.core.interface import PluginInterface
from astro_engine.adapters.base import EphemerisEngineBase
from astro_engine.models.context import Context
from astro_engine.models.location import Location
from astro_engine.models.date import Date
from astro_engine.models.date_range import DateRange
from astro_engine.models.planet import PlanetName
from astro_engine.models.events import Period
from astro_engine.models.event_repository import EventRepository

# Planets that never retrograde (Sun/Moon) or whose "retrograde" is definitional
# (the nodes are always retrograde), so a retrograde *period* is meaningless.
_SKIP = (PlanetName.SUN, PlanetName.MOON, PlanetName.RAHU, PlanetName.KETU)

# Upper bound (days) for walking to the nearest station; longer than any
# planet's synodic cycle.
_MAX_BRACKET_DAYS = 800


class RetrogradePlugin(PluginInterface):
    """Detect retrograde periods using speed sign changes and bisection refine."""

    @property
    def name(self) -> str:
        return "retrograde"

    def run(self, context: Context, ephemeris: EphemerisEngineBase) -> EventRepository:
        master_repo = EventRepository()
        planets_to_check = context.planets or list(PlanetName)

        for planet in planets_to_check:
            if planet in _SKIP:
                continue
            planet_repo = self._compute_for_single_planet(planet, context, ephemeris)
            master_repo.extend(planet_repo)
        return master_repo

    def _refine(
        self,
        planet: PlanetName,
        d1: Date,
        d2: Date,
        ephemeris: EphemerisEngineBase,
        location: Location,
        tol: float = 1e-8,
        max_iter: int = 40,
    ) -> Date:
        """Bisection for the instant where speed crosses zero within ``[d1, d2]``.

        Assumes the speed has opposite signs at ``d1`` and ``d2``.
        """
        s1 = ephemeris.get_planet_speed(planet, d1, location)
        for _ in range(max_iter):
            mid_dt = d1.to_datetime() + (d2.to_datetime() - d1.to_datetime()) / 2
            mid = Date(mid_dt)
            speed = ephemeris.get_planet_speed(planet, mid, location)

            if abs(speed) < tol:
                return mid

            if s1 * speed < 0:
                d2 = mid
            else:
                d1 = mid
                s1 = speed

        return Date(d1.to_datetime() + (d2.to_datetime() - d1.to_datetime()) / 2)

    def _find_station_bracket(
        self,
        planet: PlanetName,
        anchor: Date,
        forward: bool,
        ephemeris: EphemerisEngineBase,
        location: Location,
    ) -> Optional[Tuple[Date, Date]]:
        """Step day-by-day from ``anchor`` until the motion direction flips.

        Returns a one-day ``(earlier, later)`` bracket around the station, or
        ``None`` if no flip occurs within :data:`_MAX_BRACKET_DAYS`.
        """
        step = timedelta(days=1) if forward else timedelta(days=-1)
        initial_retro = ephemeris.get_planet_speed(planet, anchor, location) < 0

        prev = anchor
        for _ in range(_MAX_BRACKET_DAYS):
            curr = prev + step
            curr_retro = ephemeris.get_planet_speed(planet, curr, location) < 0
            if curr_retro != initial_retro:
                return (prev, curr) if forward else (curr, prev)
            prev = curr
        return None

    def _compute_for_single_planet(
        self, planet: PlanetName, context: Context, ephemeris: EphemerisEngineBase
    ) -> EventRepository:
        repo = EventRepository()
        start = context.date_range.start
        end = context.date_range.end
        location = context.location

        in_retro = ephemeris.get_planet_speed(planet, start, location) < 0

        # If already retrograde at the start, walk *backwards* to the real
        # station so the period start is accurate (previously a fixed 365-day
        # bracket could converge on the wrong crossing for fast planets).
        retrograde_start_date: Optional[Date] = None
        if in_retro:
            bracket = self._find_station_bracket(planet, start, forward=False, ephemeris=ephemeris, location=location)
            if bracket:
                retrograde_start_date = self._refine(planet, bracket[0], bracket[1], ephemeris, location)
            else:
                retrograde_start_date = start

        prev_date = start
        prev_speed = ephemeris.get_planet_speed(planet, prev_date, location)

        curr_date = start + timedelta(days=1)
        while curr_date <= end:
            curr_speed = ephemeris.get_planet_speed(planet, curr_date, location)

            if not in_retro and prev_speed >= 0 and curr_speed < 0:
                in_retro = True
                retrograde_start_date = self._refine(planet, prev_date, curr_date, ephemeris, location)

            elif in_retro and prev_speed < 0 and curr_speed >= 0:
                in_retro = False
                retrograde_end_date = self._refine(planet, prev_date, curr_date, ephemeris, location)
                if retrograde_start_date:
                    repo.add(Period(
                        event_type=self.name,
                        planet=planet,
                        date_range=DateRange(retrograde_start_date, retrograde_end_date),
                    ))
                    retrograde_start_date = None

            prev_date = curr_date
            prev_speed = curr_speed
            curr_date += timedelta(days=1)

        # Still retrograde at the end: walk *forward* to the closing station.
        if in_retro and retrograde_start_date:
            bracket = self._find_station_bracket(planet, end, forward=True, ephemeris=ephemeris, location=location)
            if bracket:
                retrograde_end_date = self._refine(planet, bracket[0], bracket[1], ephemeris, location)
            else:
                retrograde_end_date = end
            repo.add(Period(
                event_type=self.name,
                planet=planet,
                date_range=DateRange(retrograde_start_date, retrograde_end_date),
            ))

        return repo
