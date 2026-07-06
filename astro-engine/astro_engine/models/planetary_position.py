"""Immutable, richly-derived snapshot of a planet's position at an instant."""
from dataclasses import dataclass

from .date import Date
from .location import Location
from .planet import PlanetName
from .zodiac_sign import ZodiacSign
from .nakshatra import Nakshatra
from ..adapters.base import EphemerisEngineBase
from ..utils.astrometry import (
    get_rasi,
    get_nakshatra_and_pada,
    longitude_to_dms_string,
)


@dataclass(frozen=True)
class PlanetaryPosition:
    """A planet's sidereal longitude plus derived astrological properties."""

    # Inputs
    planet: PlanetName
    date: Date

    # Core calculated data
    longitude: float
    speed: float

    # Derived astrological properties
    rasi: ZodiacSign
    nakshatra: Nakshatra
    pada: int
    dms: str

    @classmethod
    def from_datetime(
        cls,
        planet: PlanetName,
        date: Date,
        location: Location,
        ephemeris: EphemerisEngineBase,
    ) -> "PlanetaryPosition":
        """Factory that queries ``ephemeris`` and derives rasi/nakshatra/pada."""
        longitude = ephemeris.get_planet_longitude(planet, date, location)
        speed = ephemeris.get_planet_speed(planet, date, location)

        rasi = get_rasi(longitude)
        nakshatra, pada = get_nakshatra_and_pada(longitude)
        dms = longitude_to_dms_string(longitude)

        return cls(
            planet=planet, date=date, longitude=longitude, speed=speed,
            rasi=rasi, nakshatra=nakshatra, pada=pada, dms=dms,
        )

    def __repr__(self) -> str:
        return (
            f"<PlanetaryPosition {self.planet.value} at {self.dms}, "
            f"Nakshatra: {self.nakshatra.value} (Pada {self.pada})>"
        )
