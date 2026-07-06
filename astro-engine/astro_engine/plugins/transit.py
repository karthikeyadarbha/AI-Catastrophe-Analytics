"""Detect rasi (sign) and nakshatra/pada ingress transits."""
from datetime import timedelta

from astro_engine.core.interface import PluginInterface
from astro_engine.adapters.base import EphemerisEngineBase
from astro_engine.models.context import Context
from astro_engine.models.location import Location
from astro_engine.models.date import Date
from astro_engine.models.planet import PlanetName
from astro_engine.models.zodiac_sign import ZodiacSign
from astro_engine.models.nakshatra import Nakshatra
from astro_engine.models.events import InstantEvent
from astro_engine.models.event_repository import EventRepository
from astro_engine.utils.astrometry import get_rasi, get_nakshatra_and_pada


class RasiTransitPlugin(PluginInterface):
    """Detect the exact instant a body enters a new rasi (zodiac sign)."""

    @property
    def name(self) -> str:
        return "rasi_transit"

    def run(self, context: Context, ephemeris: EphemerisEngineBase) -> EventRepository:
        master_repo = EventRepository()
        planets_to_check = context.planets or list(PlanetName)
        for planet in planets_to_check:
            master_repo.extend(self._compute_for_single_planet(planet, context, ephemeris))
        return master_repo

    def _refine(
        self, planet: PlanetName, d1: Date, d2: Date, target_rasi: ZodiacSign,
        ephemeris: EphemerisEngineBase, location: Location,
        tol: float = 1e-3, max_iter: int = 50,
    ) -> Date:
        """Bisection for the instant of ingress into ``target_rasi``."""
        if d1 > d2:
            d1, d2 = d2, d1

        for _ in range(max_iter):
            if (d2 - d1).total_seconds() < tol:
                break
            mid = Date(d1.to_datetime() + (d2.to_datetime() - d1.to_datetime()) / 2)
            mid_rasi = get_rasi(ephemeris.get_planet_longitude(planet, mid, location))
            if mid_rasi == target_rasi:
                d2 = mid
            else:
                d1 = mid
        return d2

    def _compute_for_single_planet(
        self, planet: PlanetName, context: Context, ephemeris: EphemerisEngineBase
    ) -> EventRepository:
        repo = EventRepository()
        start, end, location = context.date_range.start, context.date_range.end, context.location

        # The Moon crosses a sign every ~2.3 days, so sample it more finely.
        time_step = timedelta(hours=6) if planet == PlanetName.MOON else timedelta(days=1)

        prev_date = start
        prev_rasi = get_rasi(ephemeris.get_planet_longitude(planet, prev_date, location))

        curr_date = start + time_step
        while curr_date <= end:
            curr_rasi = get_rasi(ephemeris.get_planet_longitude(planet, curr_date, location))

            if prev_rasi != curr_rasi:
                transit_date = self._refine(planet, prev_date, curr_date, curr_rasi, ephemeris, location)
                repo.add(InstantEvent(
                    event_type=self.name,
                    planet=planet,
                    date=transit_date,
                    extra={"rasi_entered": curr_rasi.value, "rasi_left": prev_rasi.value},
                ))

            prev_date = curr_date
            prev_rasi = curr_rasi
            curr_date += time_step
        return repo


class NakshatraTransitPlugin(PluginInterface):
    """Detect the exact instant a body enters a new nakshatra or pada."""

    @property
    def name(self) -> str:
        return "nakshatra_transit"

    def run(self, context: Context, ephemeris: EphemerisEngineBase) -> EventRepository:
        master_repo = EventRepository()
        planets_to_check = context.planets or list(PlanetName)
        for planet in planets_to_check:
            master_repo.extend(self._compute_for_single_planet(planet, context, ephemeris))
        return master_repo

    def _refine(
        self, planet: PlanetName, d1: Date, d2: Date,
        target_nakshatra: Nakshatra, target_pada: int,
        ephemeris: EphemerisEngineBase, location: Location,
        tol: float = 1e-3, max_iter: int = 50,
    ) -> Date:
        """Bisection for the instant of ingress into the target nakshatra/pada."""
        if d1 > d2:
            d1, d2 = d2, d1

        for _ in range(max_iter):
            if (d2 - d1).total_seconds() < tol:
                break
            mid = Date(d1.to_datetime() + (d2.to_datetime() - d1.to_datetime()) / 2)
            mid_nak, mid_pada = get_nakshatra_and_pada(
                ephemeris.get_planet_longitude(planet, mid, location)
            )
            if mid_nak == target_nakshatra and mid_pada == target_pada:
                d2 = mid
            else:
                d1 = mid
        return d2

    def _compute_for_single_planet(
        self, planet: PlanetName, context: Context, ephemeris: EphemerisEngineBase
    ) -> EventRepository:
        repo = EventRepository()
        start, end, location = context.date_range.start, context.date_range.end, context.location

        time_step = timedelta(hours=1) if planet == PlanetName.MOON else timedelta(days=1)

        prev_date = start
        prev_nak, prev_pada = get_nakshatra_and_pada(
            ephemeris.get_planet_longitude(planet, prev_date, location)
        )

        curr_date = start + time_step
        while curr_date <= end:
            curr_nak, curr_pada = get_nakshatra_and_pada(
                ephemeris.get_planet_longitude(planet, curr_date, location)
            )

            if prev_nak != curr_nak or prev_pada != curr_pada:
                transit_date = self._refine(
                    planet, prev_date, curr_date, curr_nak, curr_pada, ephemeris, location
                )
                repo.add(InstantEvent(
                    event_type=self.name,
                    planet=planet,
                    date=transit_date,
                    extra={
                        "nakshatra_entered": curr_nak.value,
                        "pada_entered": curr_pada,
                        "nakshatra_left": prev_nak.value,
                        "pada_left": prev_pada,
                    },
                ))

            prev_date = curr_date
            prev_nak, prev_pada = curr_nak, curr_pada
            curr_date += time_step
        return repo
