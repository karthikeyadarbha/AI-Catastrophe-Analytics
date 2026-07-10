"""Constants for the JPL (Skyfield) backend."""
from astro_engine.models.planet import PlanetName

# Candidate Skyfield target names per planet, most-specific first. Small DE
# kernels (e.g. de421) only carry barycenters for Mars and the outer planets,
# so we fall back to the barycenter when the planet centre is absent. The
# planet/barycenter offset is far below astrological resolution.
TARGET_CANDIDATES = {
    PlanetName.SUN: ("sun",),
    PlanetName.MOON: ("moon",),
    PlanetName.MERCURY: ("mercury", "mercury barycenter"),
    PlanetName.VENUS: ("venus", "venus barycenter"),
    PlanetName.MARS: ("mars", "mars barycenter"),
    PlanetName.JUPITER: ("jupiter", "jupiter barycenter"),
    PlanetName.SATURN: ("saturn", "saturn barycenter"),
}

# The lunar nodes (Rahu/Ketu) have no ephemeris body; they are computed
# analytically from the Moon's mean node.
NODE_PLANETS = (PlanetName.RAHU, PlanetName.KETU)

# Finite-difference half-step (days) for numerical longitudinal speed.
SPEED_STEP_DAYS = 1.0 / 24.0  # one hour
