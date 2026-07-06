"""The Ascendant (Lagna): the rising sign/degree at an instant and place."""
from dataclasses import dataclass

from .date import Date
from .location import Location
from .zodiac_sign import ZodiacSign
from .nakshatra import Nakshatra
from ..utils.astrometry import (
    get_rasi,
    get_nakshatra_and_pada,
    longitude_to_dms_string,
)


@dataclass(frozen=True)
class Ascendant:
    """The Ascendant / Lagna and its derived astrological properties.

    The Ascendant is the sidereal ecliptic longitude rising on the eastern
    horizon. ``rasi`` is the *lagna rasi* (the rising sign).
    """

    date: Date
    location: Location
    longitude: float
    rasi: ZodiacSign
    nakshatra: Nakshatra
    pada: int
    dms: str

    @classmethod
    def from_longitude(cls, longitude: float, date: Date, location: Location) -> "Ascendant":
        """Build an :class:`Ascendant` from a sidereal longitude, deriving the
        rasi, nakshatra, pada and DMS string."""
        longitude = longitude % 360.0
        nakshatra, pada = get_nakshatra_and_pada(longitude)
        return cls(
            date=date,
            location=location,
            longitude=longitude,
            rasi=get_rasi(longitude),
            nakshatra=nakshatra,
            pada=pada,
            dms=longitude_to_dms_string(longitude),
        )

    def __repr__(self) -> str:
        return (
            f"<Ascendant (Lagna) at {self.dms}, "
            f"Nakshatra: {self.nakshatra.value} (Pada {self.pada})>"
        )
