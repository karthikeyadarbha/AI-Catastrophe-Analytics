"""Timezone-aware date/time wrapper used throughout the engine."""
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Union

DEFAULT_TZ = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class Date:
    """An immutable, timezone-aware instant.

    Naive datetimes are assumed to be in ``tz`` (default Asia/Kolkata).
    Equality and ordering compare the underlying instant (POSIX timestamp),
    so two ``Date`` objects in different zones are equal when they refer to
    the same moment.
    """

    dt: datetime = field(compare=False)

    def __init__(self, dt: datetime, tz: ZoneInfo = DEFAULT_TZ):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        object.__setattr__(self, "dt", dt)

    @classmethod
    def from_iso(cls, iso_str: str, tz: ZoneInfo = DEFAULT_TZ) -> "Date":
        """Build a :class:`Date` from an ISO-8601 string.

        A naive string is localised to ``tz``; a ``+00:00`` / ``Z`` offset is
        normalised to ``ZoneInfo("UTC")`` so ``dt.tzinfo`` compares cleanly.
        """
        dt = datetime.fromisoformat(iso_str)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        elif dt.utcoffset() == timedelta(0):
            # Normalise Python's fixed-offset UTC to ZoneInfo("UTC").
            dt = dt.astimezone(ZoneInfo("UTC"))

        return cls(dt)

    def __add__(self, other: timedelta) -> "Date":
        if isinstance(other, timedelta):
            return Date(self.dt + other)
        return NotImplemented

    def __sub__(self, other: Union[timedelta, "Date"]) -> Union["Date", timedelta]:
        if isinstance(other, timedelta):
            return Date(self.dt - other)
        if isinstance(other, Date):
            return self.dt - other.dt
        return NotImplemented

    def in_timezone(self, tz: ZoneInfo) -> "Date":
        """Return a new Date instance in a different timezone."""
        return Date(self.dt.astimezone(tz))

    def to_datetime(self) -> datetime:
        return self.dt

    def timestamp(self) -> float:
        return self.dt.timestamp()

    def date(self) -> "datetime.date":
        return self.dt.date()

    def time(self) -> "datetime.time":
        return self.dt.time()

    def is_same_wall_time(self, other: "Date") -> bool:
        """Returns True if local date and time match (ignores timezone)."""
        return self.dt.date() == other.dt.date() and self.dt.time() == other.dt.time()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Date) and self.timestamp() == other.timestamp()

    def __hash__(self) -> int:
        return hash(self.timestamp())

    def __lt__(self, other: "Date") -> bool:
        return self.timestamp() < other.timestamp()

    def __le__(self, other: "Date") -> bool:
        return self.timestamp() <= other.timestamp()

    def __gt__(self, other: "Date") -> bool:
        return self.timestamp() > other.timestamp()

    def __ge__(self, other: "Date") -> bool:
        return self.timestamp() >= other.timestamp()

    def __str__(self) -> str:
        return self.dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    def __repr__(self) -> str:
        return f"Date({self.dt.isoformat()})"
