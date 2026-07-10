"""Predicate object for filtering collections of events."""
from typing import Optional, Type
from .events import Event, Period, InstantEvent
from .date_range import DateRange
from .planet import PlanetName


class EventFilter:
    """Filters a collection of :class:`Event` objects by several criteria.

    All supplied criteria must match (logical AND). Omitted criteria are
    ignored.
    """

    def __init__(
        self,
        planet: Optional[PlanetName] = None,
        date_range: Optional[DateRange] = None,
        event_type: Optional[str] = None,
        event_class: Optional[Type[Event]] = None,
    ):
        self.planet = planet
        self.date_range = date_range
        self.event_type = event_type
        self.event_class = event_class

    def matches(self, event: Event) -> bool:
        """Return ``True`` if ``event`` satisfies every configured criterion."""
        if self.event_class and not isinstance(event, self.event_class):
            return False

        if self.event_type and event.event_type != self.event_type:
            return False

        if self.planet and hasattr(event, "planet") and event.planet != self.planet:
            return False

        if self.date_range:
            if isinstance(event, Period) and not self.date_range.overlaps(event.date_range):
                return False
            elif isinstance(event, InstantEvent) and not self.date_range.contains(event.date):
                return False

        return True
