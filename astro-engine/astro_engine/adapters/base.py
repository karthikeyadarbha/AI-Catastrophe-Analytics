"""The ephemeris *port*: the interface every backend must implement."""
from abc import ABC, abstractmethod

from astro_engine.models.date import Date
from astro_engine.models.planet import PlanetName
from astro_engine.models.location import Location
from astro_engine.models.motion import MotionType


class EphemerisEngineBase(ABC):
    """Abstract base class for all ephemeris engines.

    Semantics that every backend must honour so results are interchangeable:

    * ``get_planet_longitude`` returns the **sidereal** ecliptic longitude in
      degrees ``[0, 360)`` using the engine's configured ayanamsa. By default
      this is **geocentric**; backends may offer an opt-in topocentric mode.
    * ``get_planet_speed`` returns the **geocentric tropical** longitudinal
      speed in degrees/day. Positive is direct (prograde), negative is
      retrograde. This frame is the convention for defining retrograde events
      and is deliberately independent of the observer's location.
    * ``get_planet_motion`` classifies the sign of the speed.

    Ketu is defined as Rahu + 180 degrees.
    """

    @abstractmethod
    def get_planet_longitude(self, planet: PlanetName, date: Date, location: Location) -> float:
        """Return the sidereal ecliptic longitude (degrees) of ``planet``."""

    @abstractmethod
    def get_planet_motion(self, planet: PlanetName, date: Date, location: Location) -> MotionType:
        """Return whether ``planet`` is direct or retrograde."""

    @abstractmethod
    def get_planet_speed(self, planet: PlanetName, date: Date, location: Location) -> float:
        """Return the geocentric tropical longitudinal speed (degrees/day)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the engine (e.g. ``'swiss'``, ``'jpl'``)."""
