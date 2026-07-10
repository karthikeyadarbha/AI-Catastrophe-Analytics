"""In-memory collection of events with optional filtering."""
from typing import List, Optional
from .events import Event
from .event_filter import EventFilter


class EventRepository:
    """A simple in-memory collection of :class:`Event` objects."""

    def __init__(self):
        self._events: List[Event] = []

    def add(self, event: Event) -> None:
        self._events.append(event)

    def extend(self, events) -> None:
        """Add every event from an iterable (e.g. another repository)."""
        for event in events:
            self._events.append(event)

    def query(self, filt: Optional[EventFilter] = None) -> List[Event]:
        """Return events matching ``filt`` (or all events if ``filt`` is None)."""
        if filt is None:
            return list(self._events)
        return [e for e in self._events if filt.matches(e)]

    def __iter__(self):
        return iter(self._events)

    def __len__(self):
        return len(self._events)

    def __repr__(self):
        return f"EventRepository({len(self._events)} event(s))"
