"""Event value objects produced by plugins."""
from dataclasses import dataclass, field
from typing import Any

from .date import Date
from .date_range import DateRange
from .planet import PlanetName


@dataclass(frozen=True, kw_only=True)
class Event:
    """Base class for all events detected by the engine."""

    event_type: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Period(Event):
    """An event that lasts for a duration (e.g. retrograde, combustion)."""

    date_range: DateRange
    planet: PlanetName

    def __repr__(self) -> str:
        return (
            f"{self.event_type.title()} period for {self.planet.value}: "
            f"{self.date_range.start} \u2192 {self.date_range.end}"
        )


@dataclass(frozen=True)
class InstantEvent(Event):
    """An event that occurs at a single instant (e.g. a sign transit)."""

    date: Date
    planet: PlanetName

    def __repr__(self) -> str:
        return f"{self.event_type.title()} for {self.planet.value} on {self.date}"
