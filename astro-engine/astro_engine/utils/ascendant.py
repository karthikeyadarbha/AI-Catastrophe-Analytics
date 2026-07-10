"""Ascendant (Lagna) geometry: sidereal time, obliquity and the rising degree.

The **Ascendant** (Sanskrit *Lagna*, *udaya lagna*) is the ecliptic longitude of
the point rising on the eastern horizon at a given instant and place. Unlike a
planet, it depends on the observer's latitude, longitude and the local sidereal
time, so it is computed here rather than read from an ephemeris body.

Method (matches Swiss Ephemeris / drikpanchang to sub-arcminute across
1600-2400):

1. Greenwich Mean Sidereal Time from the **UT** Julian Day (IAU-1982). Deriving
   sidereal time directly from UT -- rather than routing through TT/UT1 -- is
   what keeps far-past/future dates aligned with Swiss Ephemeris.
2. Add the equation of the equinoxes (nutation) to get apparent sidereal time,
   then add the (east-positive) geographic longitude to get the local RAMC.
3. Solve the classic Ascendant equation for the tropical rising longitude.
4. Subtract the ayanamsa to obtain the sidereal (Vedic) Ascendant.

Steps 1, 3 and 4 are pure functions here; nutation for step 2 is supplied by
the caller (the JPL backend uses Skyfield's nutation series).
"""
import math

_J2000 = 2451545.0


def mean_obliquity(jd_tt: float) -> float:
    """Mean obliquity of the ecliptic in degrees (IAU 1980), TT Julian Day."""
    t = (jd_tt - _J2000) / 36525.0
    seconds = 84381.448 - 46.8150 * t - 0.00059 * t * t + 0.001813 * t * t * t
    return seconds / 3600.0


def greenwich_mean_sidereal_time(jd_ut: float) -> float:
    """Greenwich Mean Sidereal Time in degrees ``[0, 360)`` (IAU 1982).

    Evaluated from the **UT** Julian Day, which is the convention Swiss
    Ephemeris uses; this is essential for agreement at dates far from J2000.
    """
    d = jd_ut - _J2000
    t = d / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * d
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    return gmst % 360.0


def local_apparent_sidereal_time(
    jd_ut: float, longitude_east: float, equation_of_equinoxes: float = 0.0
) -> float:
    """Local Apparent Sidereal Time (RAMC) in degrees ``[0, 360)``.

    Args:
        jd_ut: UT Julian Day.
        longitude_east: Geographic longitude, east positive, degrees.
        equation_of_equinoxes: Nutation correction (deg). ``0`` yields the mean
            sidereal time, which is accurate to the ~arcsecond level.
    """
    return (
        greenwich_mean_sidereal_time(jd_ut) + equation_of_equinoxes + longitude_east
    ) % 360.0


def tropical_ascendant(ramc_deg: float, obliquity_deg: float, latitude_deg: float) -> float:
    """Tropical ecliptic longitude of the Ascendant in degrees ``[0, 360)``.

    Args:
        ramc_deg: Local (apparent) sidereal time in degrees.
        obliquity_deg: Obliquity of the ecliptic in degrees.
        latitude_deg: Geographic latitude, north positive, degrees.
    """
    ramc = math.radians(ramc_deg)
    eps = math.radians(obliquity_deg)
    phi = math.radians(latitude_deg)
    y = math.cos(ramc)
    x = -(math.sin(ramc) * math.cos(eps) + math.tan(phi) * math.sin(eps))
    return math.degrees(math.atan2(y, x)) % 360.0
