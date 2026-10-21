"""
Core domain models and entities for the celestial state machine.
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from ..constants import SWISSEPH_PLANETS, DEFAULT_TIMEZONE
from ..utils import ensure_tzaware, format_dms_str

class Planet(str, Enum):
    SUN = "Sun"
    MOON = "Moon"
    MARS = "Mars"
    MERCURY = "Mercury"
    JUPITER = "Jupiter"
    VENUS = "Venus"
    SATURN = "Saturn"
    RAHU = "Rahu"
    KETU = "Ketu"
    URANUS = "Uranus"
    NEPTUNE = "Neptune"
    PLUTO = "Pluto"

    @classmethod
    def from_string(cls, name: str) -> Planet:
        clean = name.strip().capitalize()
        for member in cls:
            if member.value.lower() == clean.lower():
                return member
        raise ValueError(f"Unknown planet '{name}'. Valid: {[p.value for p in cls]}")

class MotionState(str, Enum):
    DIRECT = "Direct"
    RETROGRADE = "Retrograde"
    STATIONARY = "Stationary"
    ACCELERATED = "Atichara"

@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime

    def __post_init__(self):
        start_tz = ensure_tzaware(self.start)
        end_tz = ensure_tzaware(self.end)
        if start_tz > end_tz:
            raise ValueError(f"start ({start_tz}) cannot be after end ({end_tz})")
        object.__setattr__(self, "start", start_tz)
        object.__setattr__(self, "end", end_tz)

    def days(self, step: int = 1):
        """Generator stepping through dates."""
        curr = self.start
        delta = timedelta(days=step)
        while curr <= self.end:
            yield curr
            curr += delta

    def contains(self, dt: datetime) -> bool:
        dt_tz = ensure_tzaware(dt)
        return self.start <= dt_tz <= self.end

    @property
    def duration_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400.0

@dataclass
class Period:
    """An interval representing an active state or event window."""
    start: datetime
    end: datetime
    event_type: str
    planet: Optional[Planet] = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.start = ensure_tzaware(self.start)
        self.end = ensure_tzaware(self.end)

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def duration_days(self) -> float:
        return self.duration_seconds / 86400.0

    def contains(self, dt: datetime) -> bool:
        dt_tz = ensure_tzaware(dt)
        return self.start <= dt_tz <= self.end

    def overlaps(self, other: Period) -> bool:
        return self.start <= other.end and other.start <= self.end

    def intersection(self, other: Period) -> Optional[Period]:
        if not self.overlaps(other):
            return None
        new_start = max(self.start, other.start)
        new_end = min(self.end, other.end)
        return Period(
            start=new_start,
            end=new_end,
            event_type=f"{self.event_type} & {other.event_type}",
            details={"source_a": self.to_dict(), "source_b": other.to_dict()}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "planet": self.planet.value if self.planet else None,
            "event_type": self.event_type,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_days": round(self.duration_days, 3),
            "details": self.details,
        }

@dataclass
class PlanetPosition:
    planet: Planet
    longitude: float
    latitude: float
    speed: float
    motion: MotionState
    sign: str
    sign_index: int           # 1 to 12
    degree_in_sign: float     # 0.0 to 30.0
    nakshatra: str
    nakshatra_index: int     # 1 to 27
    pada: int                 # 1 to 4
    dms: str
    is_combust: bool
    chara_karaka: Optional[str] = None
    altitude_deg: Optional[float] = None
    is_above_horizon: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        res = {
            "planet": self.planet.value,
            "longitude": round(self.longitude, 4),
            "latitude": round(self.latitude, 4),
            "speed_deg_per_day": round(self.speed, 4),
            "motion": self.motion.value,
            "sign": self.sign,
            "sign_index": self.sign_index,
            "degree_in_sign": round(self.degree_in_sign, 4),
            "nakshatra": self.nakshatra,
            "nakshatra_index": self.nakshatra_index,
            "pada": self.pada,
            "dms": self.dms,
            "is_combust": self.is_combust,
            "chara_karaka": self.chara_karaka,
        }
        if self.altitude_deg is not None:
            res["altitude_deg"] = round(self.altitude_deg, 2)
            res["is_above_horizon"] = self.is_above_horizon
        return res

@dataclass
class TransitionEvent:
    timestamp: datetime
    event_type: str
    planet: Planet
    description: str
    old_value: Any
    new_value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "planet": self.planet.value,
            "event_type": self.event_type,
            "description": self.description,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }
