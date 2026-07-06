"""Astro Engine: a modular Vedic-astrology ephemeris and event-detection library.

Quick start::

    from astro_engine import AstroEngine

    engine = AstroEngine(backend="swiss")          # or backend="jpl"
    pos = engine.position("Mars", "2025-01-01", (17.385, 78.4867))
    print(pos.rasi, pos.nakshatra, pos.dms)

    events = engine.find_events(
        start="2025-01-01", end="2025-12-31",
        location=(17.385, 78.4867),
        plugins=["retrograde", "combustion"],
    )
    for e in events:
        print(e)
"""
from .facade import AstroEngine

# Core orchestration
from .core.engine import EngineManager
from .core.registry import PluginRegistry
from .core.interface import PluginInterface
from .core.exceptions import AstroError, EphemerisError, PluginError

# Ephemeris port
from .adapters.base import EphemerisEngineBase

# Domain models
from .models.planet import PlanetName
from .models.motion import MotionType
from .models.zodiac_sign import ZodiacSign
from .models.nakshatra import Nakshatra
from .models.location import Location
from .models.date import Date
from .models.date_range import DateRange
from .models.context import Context
from .models.events import Event, Period, InstantEvent
from .models.event_filter import EventFilter
from .models.event_repository import EventRepository
from .models.planetary_position import PlanetaryPosition

# Utilities
from .utils.ayanamsa import lahiri_ayanamsa, lahiri_ayanamsa_at
from .utils.astrometry import get_rasi, get_nakshatra_and_pada, longitude_to_dms_string

__version__ = "0.1.0"

__all__ = [
    "AstroEngine",
    "EngineManager",
    "PluginRegistry",
    "PluginInterface",
    "AstroError",
    "EphemerisError",
    "PluginError",
    "EphemerisEngineBase",
    "PlanetName",
    "MotionType",
    "ZodiacSign",
    "Nakshatra",
    "Location",
    "Date",
    "DateRange",
    "Context",
    "Event",
    "Period",
    "InstantEvent",
    "EventFilter",
    "EventRepository",
    "PlanetaryPosition",
    "lahiri_ayanamsa",
    "lahiri_ayanamsa_at",
    "get_rasi",
    "get_nakshatra_and_pada",
    "longitude_to_dms_string",
    "__version__",
]
