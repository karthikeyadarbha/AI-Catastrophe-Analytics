"""Small datetime helpers."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional


def parse_datetime(
    dt_str: str,
    fmt: str = "%Y-%m-%d %H:%M:%S",
    tz: Optional[str] = None,
) -> datetime:
    """Parse a datetime string into a timezone-aware ``datetime``.

    If ``tz`` is given it is applied as the local zone; otherwise UTC.
    """
    dt = datetime.strptime(dt_str, fmt)
    if tz:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    else:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def to_utc(dt: datetime) -> datetime:
    """Convert any datetime (aware or naive) to UTC. Naive is assumed UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime into a string."""
    return dt.strftime(fmt)
