"""Constants mapping engine enums to Swiss Ephemeris identifiers."""
import swisseph as swe

# Named sidereal modes (ayanamsas) supported by the Swiss backend.
SIDEREAL_MODE_MAP = {
    "Lahiri": swe.SIDM_LAHIRI,
    "Raman": swe.SIDM_RAMAN,
    "KP": getattr(swe, "SIDM_KRISHNAMURTI", swe.SIDM_LAHIRI),
    "FaganBradley": swe.SIDM_FAGAN_BRADLEY,
    "default": swe.SIDM_LAHIRI,
}

# Map our PlanetName values to Swiss Ephemeris planet ids.
# Rahu/Ketu both derive from the mean lunar node (Ketu = node + 180).
SWISSEPH_PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE,
    "Ketu": swe.MEAN_NODE,
}
