"""The plugin contract."""
from abc import ABC, abstractmethod

from astro_engine.models.context import Context
from astro_engine.models.event_repository import EventRepository
from astro_engine.adapters.base import EphemerisEngineBase


class PluginInterface(ABC):
    """Contract for all event-detection plugins.

    Each plugin is a self-contained module that identifies specific
    astronomical events (retrogrades, combustion, transits, ...). By adhering
    to this interface, :class:`~astro_engine.core.engine.EngineManager` can
    discover, load and run any plugin without knowing its internals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A unique, machine-readable name (e.g. ``'retrograde'``)."""

    @abstractmethod
    def run(self, context: Context, ephemeris: EphemerisEngineBase) -> EventRepository:
        """Detect events over ``context`` using ``ephemeris`` and return them."""
