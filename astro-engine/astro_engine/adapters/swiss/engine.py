"""Swiss Ephemeris backend (``pyswisseph``)."""
import threading
from datetime import timezone
from typing import Dict, Tuple

import swisseph as swe

from astro_engine.adapters.base import EphemerisEngineBase
from astro_engine.models.date import Date
from astro_engine.models.planet import PlanetName
from astro_engine.models.location import Location
from astro_engine.models.motion import MotionType
from astro_engine.core.exceptions import EphemerisError
from astro_engine.adapters.swiss.constants import SIDEREAL_MODE_MAP, SWISSEPH_PLANETS

# Swiss Ephemeris keeps sidereal mode / topocentre in *global* state, so guard
# calls with a module-level lock to stay correct under multi-threading.
_SWE_LOCK = threading.RLock()


class SwissEphemerisEngine(EphemerisEngineBase):
    """Adapter over the Swiss Ephemeris.

    Longitudes are sidereal (using ``sidereal_mode``); speeds are geocentric
    tropical. Longitudes are geocentric by default; pass ``topocentric=True``
    to correct for the observer's position (matters mostly for the Moon).

    Args:
        sidereal_mode: Ayanamsa name (see ``SIDEREAL_MODE_MAP``).
        ephe_path: Directory of Swiss Ephemeris data files (``*.se1``). If
            omitted, the built-in Moshier ephemeris is used (accurate to a few
            arcseconds for modern dates and needs no data files).
        topocentric: If ``True``, longitudes are topocentric.
    """

    def __init__(
        self,
        sidereal_mode: str = "Lahiri",
        ephe_path: str = None,
        topocentric: bool = False,
    ):
        super().__init__()
        if ephe_path:
            swe.set_ephe_path(ephe_path)
        self._sidereal_mode = sidereal_mode
        self._sid_flag = SIDEREAL_MODE_MAP.get(sidereal_mode, swe.SIDM_LAHIRI)
        self.topocentric = topocentric
        # Cache raw Swiss results keyed by (planet_id, jd, flags, topo-signature).
        self._cache: Dict[Tuple, tuple] = {}

    @property
    def name(self) -> str:
        return "swiss"

    def _get_julian_day_utc(self, date: Date) -> float:
        """Convert a :class:`Date` to the UTC Julian Day swisseph expects."""
        dt_utc = date.dt.astimezone(timezone.utc)
        return swe.utc_to_jd(
            dt_utc.year, dt_utc.month, dt_utc.day,
            dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond / 1e6,
            swe.GREG_CAL,
        )[1]

    def _calc(self, planet_id: int, jd: float, flags: int, topo=None) -> tuple:
        key = (planet_id, round(jd, 10), flags, topo)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        with _SWE_LOCK:
            swe.set_sid_mode(self._sid_flag)
            if topo is not None:
                swe.set_topo(topo[0], topo[1], topo[2])
            res, err = swe.calc_ut(jd, planet_id, flags)

        if isinstance(res, int) and res < 0:
            raise EphemerisError(f"Swiss Ephemeris error for id {planet_id}: {err}")

        self._cache[key] = res
        return res

    def get_planet_longitude(self, planet: PlanetName, date: Date, location: Location) -> float:
        """Sidereal ecliptic longitude in degrees ``[0, 360)``."""
        jd = self._get_julian_day_utc(date)
        planet_id = SWISSEPH_PLANETS[planet.value]

        flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
        topo = None
        if self.topocentric:
            flags |= swe.FLG_TOPOCTR
            topo = (location.longitude, location.latitude, location.elevation)

        res = self._calc(planet_id, jd, flags, topo)
        longitude = res[0]
        if planet == PlanetName.KETU:
            longitude += 180.0
        return longitude % 360.0

    def get_planet_speed(self, planet: PlanetName, date: Date, location: Location) -> float:
        """Geocentric tropical longitudinal speed (degrees/day)."""
        jd = self._get_julian_day_utc(date)
        planet_id = SWISSEPH_PLANETS[planet.value]

        # Tropical (no FLG_SIDEREAL) and geocentric (no FLG_TOPOCTR / set_topo).
        flags = swe.FLG_SWIEPH | swe.FLG_SPEED
        res = self._calc(planet_id, jd, flags)
        return res[3]

    def get_planet_motion(self, planet: PlanetName, date: Date, location: Location) -> MotionType:
        """Direct or retrograde, from the sign of the geocentric speed."""
        speed = self.get_planet_speed(planet, date, location)
        return MotionType.DIRECT if speed >= 0 else MotionType.RETROGRADE

    def get_ascendant(self, date: Date, location: Location) -> float:
        """Sidereal Ascendant (Lagna) longitude in degrees ``[0, 360)``.

        Uses Swiss Ephemeris house computation with the sidereal flag; the
        house system is irrelevant to the Ascendant, so Placidus is used.
        """
        jd = self._get_julian_day_utc(date)
        with _SWE_LOCK:
            swe.set_sid_mode(self._sid_flag)
            _, ascmc = swe.houses_ex(
                jd, location.latitude, location.longitude, b"P", swe.FLG_SIDEREAL
            )
        return ascmc[0] % 360.0
