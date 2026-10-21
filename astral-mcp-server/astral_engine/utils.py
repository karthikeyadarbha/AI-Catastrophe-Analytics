"""
Utility functions for celestial coordinates, time conversions, and math.
"""
from __future__ import annotations
import math
import datetime
from dateutil import parser as dt_parser
import swisseph as swe
from .constants import UTC, DEFAULT_TIMEZONE

def parse_datetime(dt_input: str | datetime.datetime) -> datetime.datetime:
    """Parse any ISO string or datetime object into a timezone-aware datetime (UTC by default)."""
    if isinstance(dt_input, datetime.datetime):
        if dt_input.tzinfo is None:
            return dt_input.replace(tzinfo=DEFAULT_TIMEZONE)
        return dt_input
    
    dt = dt_parser.isoparse(str(dt_input))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TIMEZONE)
    return dt

def ensure_tzaware(dt: datetime.datetime, default_tz: datetime.tzinfo = DEFAULT_TIMEZONE) -> datetime.datetime:
    """Ensure datetime is timezone-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=default_tz)
    return dt

def to_jd(dt: datetime.datetime) -> float:
    """Convert any timezone-aware datetime to Julian Day (UT)."""
    if dt.tzinfo is not UTC:
        dt = dt.astimezone(UTC)
    frac_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3.6e9
    return float(swe.julday(dt.year, dt.month, dt.day, frac_hour))

def jd_to_datetime(jd: float, tz: datetime.tzinfo = UTC) -> datetime.datetime:
    """Convert Julian Day back to timezone-aware datetime."""
    year, month, day, frac = swe.revjul(jd)
    hours_total = frac * 24.0
    h = int(hours_total)
    mins_total = (hours_total - h) * 60.0
    m = int(mins_total)
    secs_total = (mins_total - m) * 60.0
    s = int(secs_total)
    micro = int(round((secs_total - s) * 1e6))
    if micro >= 1000000:
        micro = 999999
    dt_utc = datetime.datetime(year, month, day, h, m, s, micro, tzinfo=UTC)
    return dt_utc.astimezone(tz)

def minimal_angular_diff(a1: float, a2: float) -> float:
    """Smallest separation between two circular angles on [0, 360)."""
    diff = abs(a2 - a1) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff

def format_dms(deg: float) -> tuple[int, int, float]:
    """Convert decimal degrees to (degrees, minutes, seconds)."""
    deg_norm = deg % 360.0
    d = int(deg_norm)
    mins_float = (deg_norm - d) * 60.0
    m = int(mins_float)
    s = (mins_float - m) * 60.0
    return (d, m, round(s, 2))

def format_dms_str(deg: float) -> str:
    """Return formatted string like 15° 24′ 36.12″."""
    d, m, s = format_dms(deg)
    return f"{d:02d}° {m:02d}′ {s:05.2f}″"
