"""An inclusive range of dates, iterable day by day."""
from dataclasses import dataclass
from typing import Iterator
from datetime import timedelta
from .date import Date


@dataclass(frozen=True)
class DateRange:
    """An inclusive ``[start, end]`` interval of :class:`Date` instants."""

    start: Date
    end: Date

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("Start date must be on or before end date")

    def days(self) -> int:
        """Number of calendar days spanned (inclusive)."""
        return (self.end.dt.date() - self.start.dt.date()).days + 1

    def contains(self, date: Date) -> bool:
        return self.start.dt <= date.dt <= self.end.dt

    def overlaps(self, other: "DateRange") -> bool:
        return self.start.dt <= other.end.dt and self.end.dt >= other.start.dt

    def __iter__(self) -> Iterator[Date]:
        current = self.start.dt
        while current <= self.end.dt:
            yield Date(current, tz=current.tzinfo)
            current += timedelta(days=1)

    def __str__(self):
        return f"{self.start} \u2192 {self.end}"

    def __repr__(self):
        return f"DateRange(start={self.start!r}, end={self.end!r})"
