"""Convert a sidereal longitude into rasi, nakshatra, pada and a DMS string."""
from typing import Tuple

from astro_engine.models.zodiac_sign import ZodiacSign
from astro_engine.models.nakshatra import Nakshatra

# Angular spans (degrees)
NAKSHATRA_SPAN = 360 / 27      # 13.333... degrees
PADA_SPAN = NAKSHATRA_SPAN / 4  # 3.333... degrees
RASI_SPAN = 30                  # 30 degrees

# Ordered lists for indexing (enums are declared in zodiacal order).
ZODIAC_SIGNS = list(ZodiacSign)
NAKSHATRAS = list(Nakshatra)


def get_rasi(longitude: float) -> ZodiacSign:
    """Return the rasi (zodiac sign) for a sidereal longitude."""
    norm_lon = longitude % 360
    sign_index = int(norm_lon / RASI_SPAN)
    return ZODIAC_SIGNS[sign_index]


def get_nakshatra_and_pada(longitude: float) -> Tuple[Nakshatra, int]:
    """Return the (nakshatra, pada) for a sidereal longitude. Pada is 1-4."""
    norm_lon = longitude % 360

    nakshatra_index = int(norm_lon / NAKSHATRA_SPAN)
    nakshatra = NAKSHATRAS[nakshatra_index]

    lon_within_nakshatra = norm_lon % NAKSHATRA_SPAN
    pada_index = int(lon_within_nakshatra / PADA_SPAN)
    pada = pada_index + 1  # Padas are 1-indexed

    return nakshatra, pada


def longitude_to_dms_string(longitude: float) -> str:
    """Format a longitude as ``D° MM' SS" (Rasi)`` within its sign."""
    norm_lon = longitude % 360

    # Degrees within the current sign (0-30), which is the conventional display.
    deg_in_sign = norm_lon % RASI_SPAN
    degrees = int(deg_in_sign)
    minutes_float = (deg_in_sign - degrees) * 60
    minutes = int(minutes_float)
    seconds_float = (minutes_float - minutes) * 60
    seconds = int(round(seconds_float))

    if seconds == 60:
        minutes += 1
        seconds = 0
    if minutes == 60:
        degrees += 1
        minutes = 0

    sign_index = int(norm_lon / RASI_SPAN)
    sign = ZODIAC_SIGNS[sign_index].value

    return f"{degrees}\u00b0 {minutes:02d}' {seconds:02d}\" ({sign})"
