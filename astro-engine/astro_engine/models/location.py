"""Geographic observer location."""
from dataclasses import dataclass
from zoneinfo import ZoneInfo

DEFAULT_ZONE = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class Location:
    """An immutable geographic location for topocentric calculations.

    Attributes:
        latitude: Degrees north of the equator, in ``[-90, 90]``.
        longitude: Degrees east of Greenwich, in ``[-180, 180]``.
        elevation: Height above sea level in metres.
        timezone: The local :class:`zoneinfo.ZoneInfo`. Defaults to Asia/Kolkata.
    """

    latitude: float
    longitude: float
    elevation: float = 0.0
    timezone: ZoneInfo = DEFAULT_ZONE

    def __post_init__(self):
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")
        if not isinstance(self.timezone, ZoneInfo):
            raise TypeError("timezone must be a zoneinfo.ZoneInfo object")

    def __str__(self):
        return f"({self.latitude}, {self.longitude}, {self.timezone.key})"

    def __repr__(self):
        return (
            f"Location(latitude={self.latitude}, longitude={self.longitude}, "
            f"elevation={self.elevation}, timezone={self.timezone.key})"
        )
