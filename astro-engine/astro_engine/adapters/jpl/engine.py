"""JPL DE ephemeris backend built on Skyfield + jplephem.

Positions come from a NASA/JPL Development Ephemeris (default ``de421.bsp``)
via Skyfield. Sidereal longitudes are produced by taking the apparent ecliptic
longitude *of date* (tropical) and subtracting the Lahiri ayanamsa, which
reproduces the Swiss backend to sub-arcsecond accuracy (see the parity test).

The lunar nodes Rahu/Ketu are not ephemeris bodies; they are computed from the
Moon's mean node using the standard polynomial (Meeus, *Astronomical
Algorithms*).
"""
import threading
from datetime import timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from astro_engine.adapters.base import EphemerisEngineBase
from astro_engine.models.date import Date
from astro_engine.models.planet import PlanetName
from astro_engine.models.location import Location
from astro_engine.models.motion import MotionType
from astro_engine.core.exceptions import EphemerisError
from astro_engine.utils.ayanamsa import get_ayanamsa, julian_day_ut
from astro_engine.adapters.jpl.constants import (
    TARGET_CANDIDATES,
    NODE_PLANETS,
    SPEED_STEP_DAYS,
)

_DEFAULT_CACHE = Path.home() / ".astro_engine" / "ephemeris"


class JplEphemerisEngine(EphemerisEngineBase):
    """Adapter over a JPL DE ephemeris using Skyfield.

    Args:
        sidereal_mode: Ayanamsa name. Only ``"Lahiri"`` is implemented by this
            pure-Python backend (use ``backend='swiss'`` for others).
        ephemeris: A DE kernel name (downloaded on first use, e.g.
            ``"de421.bsp"``, ``"de440s.bsp"``) or a path to an existing
            ``.bsp`` file.
        cache_dir: Directory to store/lookup downloaded kernels. Defaults to
            ``~/.astro_engine/ephemeris``.
        topocentric: If ``True``, longitudes are topocentric (observer at the
            given location) rather than geocentric.
    """

    def __init__(
        self,
        sidereal_mode: str = "Lahiri",
        ephemeris: str = "de421.bsp",
        cache_dir: Optional[str] = None,
        topocentric: bool = False,
    ):
        super().__init__()
        try:
            from skyfield.api import Loader, load_file, wgs84
            from skyfield.framelib import ecliptic_frame
        except ImportError as e:  # pragma: no cover - dependency guard
            raise EphemerisError(
                "The JPL backend requires 'skyfield'. Install with "
                "'pip install astro-engine[jpl]'."
            ) from e

        self._sidereal_mode = sidereal_mode
        self.topocentric = topocentric
        self._wgs84 = wgs84
        self._ecliptic_frame = ecliptic_frame

        # Load the ephemeris: an existing file path is used directly, otherwise
        # the name is downloaded into cache_dir on first use.
        eph_path = Path(ephemeris)
        if eph_path.is_file():
            self._eph = load_file(str(eph_path))
            self._loader = Loader(str(cache_dir or _DEFAULT_CACHE))
        else:
            cache = Path(cache_dir or _DEFAULT_CACHE)
            cache.mkdir(parents=True, exist_ok=True)
            self._loader = Loader(str(cache))
            self._eph = self._loader.load(ephemeris)

        self._ts = self._loader.timescale()
        self._earth = self._eph["earth"]
        self._targets: Dict[PlanetName, object] = {}
        # Cache of-date tropical ecliptic longitude by (planet, tt_jd, topo-sig).
        self._lon_cache: Dict[Tuple, float] = {}
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "jpl"

    # ------------------------------------------------------------------ #
    # Target resolution
    # ------------------------------------------------------------------ #
    def _target(self, planet: PlanetName):
        cached = self._targets.get(planet)
        if cached is not None:
            return cached
        for candidate in TARGET_CANDIDATES[planet]:
            try:
                target = self._eph[candidate]
            except (KeyError, ValueError):
                continue
            self._targets[planet] = target
            return target
        raise EphemerisError(
            f"No ephemeris body for {planet.value} in this kernel. "
            f"Tried: {TARGET_CANDIDATES[planet]}."
        )

    # ------------------------------------------------------------------ #
    # Core geometry
    # ------------------------------------------------------------------ #
    def _observer(self, location: Location):
        if self.topocentric:
            topos = self._wgs84.latlon(
                location.latitude, location.longitude, elevation_m=location.elevation
            )
            return self._earth + topos
        return self._earth

    def _geo_tropical_longitude(self, planet: PlanetName, tt_jd: float) -> float:
        """Geocentric apparent ecliptic-of-date (tropical) longitude, degrees."""
        key = (planet, round(tt_jd, 9), None)
        cached = self._lon_cache.get(key)
        if cached is not None:
            return cached

        with self._lock:
            t = self._ts.tt_jd(tt_jd)
            astrometric = self._earth.at(t).observe(self._target(planet)).apparent()
            _, lon, _ = astrometric.frame_latlon(self._ecliptic_frame)
            value = lon.degrees % 360.0

        self._lon_cache[key] = value
        return value

    def _topo_tropical_longitude(self, planet: PlanetName, tt_jd: float, location: Location) -> float:
        """Topocentric apparent ecliptic-of-date (tropical) longitude, degrees."""
        topo_sig = (
            round(location.latitude, 6),
            round(location.longitude, 6),
            round(location.elevation, 3),
        )
        key = (planet, round(tt_jd, 9), topo_sig)
        cached = self._lon_cache.get(key)
        if cached is not None:
            return cached

        with self._lock:
            t = self._ts.tt_jd(tt_jd)
            observer = self._observer(location)
            astrometric = observer.at(t).observe(self._target(planet)).apparent()
            _, lon, _ = astrometric.frame_latlon(self._ecliptic_frame)
            value = lon.degrees % 360.0

        self._lon_cache[key] = value
        return value

    def _position_longitude(self, planet: PlanetName, tt_jd: float, location: Location) -> float:
        """Tropical longitude for chart positions (topocentric if enabled)."""
        if self.topocentric:
            return self._topo_tropical_longitude(planet, tt_jd, location)
        return self._geo_tropical_longitude(planet, tt_jd)


    def _time_from_date(self, date: Date):
        dt_utc = date.dt.astimezone(timezone.utc)
        return self._ts.from_datetime(dt_utc), dt_utc

    # ------------------------------------------------------------------ #
    # Mean lunar node (Rahu/Ketu)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mean_node_tropical(tt_jd: float) -> float:
        """Moon's mean ascending node, tropical of date (degrees). Meeus."""
        t = (tt_jd - 2451545.0) / 36525.0
        omega = (
            125.0445479
            - 1934.1362891 * t
            + 0.0020754 * t * t
            + t ** 3 / 467441.0
            - t ** 4 / 60616000.0
        )
        return omega % 360.0

    @staticmethod
    def _mean_node_speed(tt_jd: float) -> float:
        """d(mean node)/dt in degrees/day (always negative -> retrograde)."""
        t = (tt_jd - 2451545.0) / 36525.0
        dodt_per_century = (
            -1934.1362891
            + 2 * 0.0020754 * t
            + 3 * t * t / 467441.0
            - 4 * t ** 3 / 60616000.0
        )
        return dodt_per_century / 36525.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_planet_longitude(self, planet: PlanetName, date: Date, location: Location) -> float:
        """Sidereal ecliptic longitude in degrees ``[0, 360)``."""
        t, dt_utc = self._time_from_date(date)
        jd_ut = julian_day_ut(dt_utc)
        ayan = get_ayanamsa(self._sidereal_mode, jd_ut)

        if planet in NODE_PLANETS:
            tropical = self._mean_node_tropical(t.tt)
            sidereal = (tropical - ayan) % 360.0
            if planet == PlanetName.KETU:
                sidereal = (sidereal + 180.0) % 360.0
            return sidereal

        tropical = self._position_longitude(planet, t.tt, location)
        return (tropical - ayan) % 360.0

    def get_planet_speed(self, planet: PlanetName, date: Date, location: Location) -> float:
        """Geocentric tropical longitudinal speed (degrees/day)."""
        t, _ = self._time_from_date(date)

        if planet in NODE_PLANETS:
            # Both nodes share the (retrograde) mean-node rate.
            return self._mean_node_speed(t.tt)

        # Central finite difference on the *geocentric* tropical longitude,
        # independent of the observer location (matches the port's semantics).
        h = SPEED_STEP_DAYS
        lon_minus = self._geo_tropical_longitude(planet, t.tt - h)
        lon_plus = self._geo_tropical_longitude(planet, t.tt + h)
        delta = ((lon_plus - lon_minus + 180.0) % 360.0) - 180.0
        return delta / (2.0 * h)


    def get_planet_motion(self, planet: PlanetName, date: Date, location: Location) -> MotionType:
        """Direct or retrograde, from the sign of the geocentric speed."""
        speed = self.get_planet_speed(planet, date, location)
        return MotionType.DIRECT if speed >= 0 else MotionType.RETROGRADE
