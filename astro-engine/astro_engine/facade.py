"""High-level, batteries-included entry point: :class:`AstroEngine`."""
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Union
from zoneinfo import ZoneInfo

from .core.engine import EngineManager
from .models.context import Context
from .models.date import Date
from .models.date_range import DateRange
from .models.location import Location
from .models.planet import PlanetName
from .models.planetary_position import PlanetaryPosition
from .models.lagna import Ascendant
from .models.event_repository import EventRepository

# Types accepted by the convenience API.
WhenLike = Union[str, datetime, Date]
LocationLike = Union[Location, Sequence[float]]
PlanetLike = Union[PlanetName, str]
TzLike = Union[str, ZoneInfo, None]


class AstroEngine:
    """A friendly wrapper over :class:`EngineManager` and the domain models.

    Example:
        >>> engine = AstroEngine(backend="jpl")
        >>> events = engine.find_events(
        ...     start="2025-01-01", end="2025-12-31",
        ...     location=(17.385, 78.4867),
        ...     plugins=["retrograde"],
        ... )
        >>> pos = engine.position("Mars", "2025-01-01", (17.385, 78.4867))
        >>> print(pos.rasi, pos.dms)
    """

    def __init__(
        self,
        backend: str = "swiss",
        *,
        ayanamsa: str = "Lahiri",
        topocentric: bool = False,
        **backend_kwargs,
    ):
        """Create an engine.

        Args:
            backend: ``"swiss"`` (default, no download) or ``"jpl"`` (Skyfield;
                downloads a DE kernel on first use).
            ayanamsa: Sidereal mode. ``"Lahiri"`` works on both backends.
            topocentric: Compute longitudes topocentrically (observer-based).
            **backend_kwargs: Extra backend options, e.g. ``ephe_path`` (swiss)
                or ``ephemeris`` / ``cache_dir`` (jpl).
        """
        self._manager = EngineManager(
            backend,
            sidereal_mode=ayanamsa,
            topocentric=topocentric,
            **backend_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        """Name of the active ephemeris backend."""
        return self._manager.ephemeris_engine.name

    @property
    def ephemeris(self):
        """The underlying :class:`EphemerisEngineBase` instance."""
        return self._manager.ephemeris_engine

    @property
    def plugins(self) -> List[str]:
        """Names of all available plugins."""
        return self._manager.plugin_registry.list_available_plugins()

    @staticmethod
    def available_backends() -> List[str]:
        return EngineManager.available_backends()

    # ------------------------------------------------------------------ #
    # Event detection
    # ------------------------------------------------------------------ #
    def find_events(
        self,
        start: WhenLike,
        end: WhenLike,
        location: LocationLike,
        *,
        plugins: Optional[Iterable[str]] = None,
        planets: Optional[Iterable[PlanetLike]] = None,
        tz: TzLike = None,
    ) -> EventRepository:
        """Scan ``[start, end]`` and return every detected event.

        Args:
            start, end: ISO strings, ``datetime`` or :class:`Date`.
            location: a :class:`Location` or a ``(lat, lon[, elev])`` sequence.
            plugins: plugin names to run (default: all).
            planets: subset of planets (names or :class:`PlanetName`).
            tz: zone for naive/str inputs; defaults to the location's timezone.
        """
        loc = self._coerce_location(location)
        zone = self._coerce_tz(tz, loc)
        date_range = DateRange(
            self._coerce_date(start, zone),
            self._coerce_date(end, zone),
        )
        plugin_configs = {name: True for name in plugins} if plugins else {}
        context = Context(
            location=loc,
            date_range=date_range,
            planets=self._coerce_planets(planets),
            plugin_configs=plugin_configs,
        )
        return self._manager.run(context)

    # ------------------------------------------------------------------ #
    # Point queries
    # ------------------------------------------------------------------ #
    def position(self, planet: PlanetLike, when: WhenLike, location: LocationLike, *, tz: TzLike = None) -> PlanetaryPosition:
        """Full :class:`PlanetaryPosition` (longitude, rasi, nakshatra, pada...)."""
        loc = self._coerce_location(location)
        zone = self._coerce_tz(tz, loc)
        return PlanetaryPosition.from_datetime(
            self._coerce_planet(planet),
            self._coerce_date(when, zone),
            loc,
            self._manager.ephemeris_engine,
        )

    def positions(
        self,
        when: WhenLike,
        location: LocationLike,
        planets: Optional[Iterable[PlanetLike]] = None,
        *,
        tz: TzLike = None,
    ) -> List[PlanetaryPosition]:
        """Positions of many planets at one instant (all nine by default)."""
        planet_list = self._coerce_planets(planets) or list(PlanetName)
        return [self.position(p, when, location, tz=tz) for p in planet_list]

    def longitude(self, planet: PlanetLike, when: WhenLike, location: LocationLike, *, tz: TzLike = None) -> float:
        """Sidereal longitude (degrees) of ``planet`` at ``when``."""
        loc = self._coerce_location(location)
        zone = self._coerce_tz(tz, loc)
        return self._manager.ephemeris_engine.get_planet_longitude(
            self._coerce_planet(planet), self._coerce_date(when, zone), loc
        )

    def lagna(self, when: WhenLike, location: LocationLike, *, tz: TzLike = None) -> Ascendant:
        """The Ascendant (Lagna) at ``when`` and ``location``.

        Returns an :class:`Ascendant` with the sidereal longitude, rising rasi,
        nakshatra, pada and a DMS string. Unlike a planet, the Ascendant depends
        on the observer's latitude/longitude and the local sidereal time.
        """
        loc = self._coerce_location(location)
        zone = self._coerce_tz(tz, loc)
        date = self._coerce_date(when, zone)
        longitude = self._manager.ephemeris_engine.get_ascendant(date, loc)
        return Ascendant.from_longitude(longitude, date, loc)

    #: Alias for :meth:`lagna`.
    ascendant = lagna

    def ascendant_longitude(self, when: WhenLike, location: LocationLike, *, tz: TzLike = None) -> float:
        """Sidereal Ascendant (Lagna) longitude in degrees only."""
        loc = self._coerce_location(location)
        zone = self._coerce_tz(tz, loc)
        return self._manager.ephemeris_engine.get_ascendant(
            self._coerce_date(when, zone), loc
        )

    # ------------------------------------------------------------------ #
    # Coercion helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_tz(tz: TzLike, location: Location) -> ZoneInfo:
        if tz is None:
            return location.timezone
        return ZoneInfo(tz) if isinstance(tz, str) else tz

    @staticmethod
    def _coerce_location(location: LocationLike) -> Location:
        if isinstance(location, Location):
            return location
        seq = list(location)
        if len(seq) == 2:
            return Location(seq[0], seq[1])
        if len(seq) >= 3:
            return Location(seq[0], seq[1], seq[2])
        raise ValueError("location must be a Location or a (lat, lon[, elev]) sequence")

    @staticmethod
    def _coerce_date(when: WhenLike, zone: ZoneInfo) -> Date:
        if isinstance(when, Date):
            return when
        if isinstance(when, datetime):
            return Date(when, tz=zone)
        if isinstance(when, str):
            return Date.from_iso(when, tz=zone)
        raise TypeError(f"Unsupported date type: {type(when)!r}")

    @staticmethod
    def _coerce_planet(planet: PlanetLike) -> PlanetName:
        return planet if isinstance(planet, PlanetName) else PlanetName(planet)

    @classmethod
    def _coerce_planets(cls, planets: Optional[Iterable[PlanetLike]]) -> Optional[List[PlanetName]]:
        if planets is None:
            return None
        return [cls._coerce_planet(p) for p in planets]
