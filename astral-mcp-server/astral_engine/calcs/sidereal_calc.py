"""
High-performance Swiss Ephemeris sidereal and topocentric calculation engine.
"""
from __future__ import annotations
import math
from datetime import datetime
import swisseph as swe
from .abstract_calc import AbstractCalc
from ..core.models import Planet, PlanetPosition, MotionState
from ..constants import (
    SWISSEPH_PLANETS, ZODIAC_SIGNS, NAKSHATRAS,
    COMBUSTION_LIMITS, AYANAMSA_MODES, DEFAULT_AYANAMSA
)
from ..utils import to_jd, minimal_angular_diff, format_dms_str

class SiderealCalc(AbstractCalc):
    def __init__(self, ayanamsa: str = DEFAULT_AYANAMSA):
        self.ayanamsa_name = ayanamsa.lower()
        self.ayanamsa_id = AYANAMSA_MODES.get(self.ayanamsa_name, swe.SIDM_LAHIRI)
        self.is_sidereal = (self.ayanamsa_id >= 0)
        if self.is_sidereal:
            swe.set_sid_mode(self.ayanamsa_id, 0, 0)

    def _get_raw_calc(self, jd: float, planet: Planet, topocentric: bool = False, lat: float = 0.0, lon: float = 0.0, elev: float = 0.0) -> tuple[float, float, float]:
        """Returns (longitude, latitude, speed_deg_per_day)."""
        if topocentric and (lat != 0.0 or lon != 0.0):
            swe.set_topo(lon, lat, elev)
            flags = swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_TOPOCTR
        else:
            flags = swe.FLG_SWIEPH | swe.FLG_SPEED

        if self.is_sidereal:
            flags |= swe.FLG_SIDEREAL

        p_name = planet.value
        if p_name == "Ketu":
            rahu_code = SWISSEPH_PLANETS["Rahu"]
            res, _ = swe.calc_ut(jd, rahu_code, flags)
            lon_sid = (res[0] + 180.0) % 360.0
            lat_sid = -res[1]
            speed_sid = -res[3]
            return (lon_sid, lat_sid, speed_sid)

        p_code = SWISSEPH_PLANETS.get(p_name)
        if p_code is None:
            raise ValueError(f"Unknown planet code for {planet}")

        res, _ = swe.calc_ut(jd, p_code, flags)
        lon_sid = res[0] % 360.0
        lat_sid = res[1]
        speed_sid = res[3]
        return (lon_sid, lat_sid, speed_sid)

    def get_motion_state(self, speed: float) -> MotionState:
        if abs(speed) < 0.005:
            return MotionState.STATIONARY
        elif speed < 0:
            return MotionState.RETROGRADE
        return MotionState.DIRECT

    def is_planet_combust(self, planet: Planet, planet_lon: float, sun_lon: float, is_retro: bool) -> bool:
        if planet in (Planet.SUN, Planet.RAHU, Planet.KETU):
            return False
        limits = COMBUSTION_LIMITS.get(planet.value)
        if not limits:
            return False
        threshold = limits["retro"] if is_retro else limits["direct"]
        sep = minimal_angular_diff(planet_lon, sun_lon)
        return sep < threshold

    def get_planet_position(self, planet: Planet, dt: datetime, lat: float = 0.0, lon: float = 0.0, elev: float = 0.0) -> PlanetPosition:
        jd = to_jd(dt)
        has_observer = (lat != 0.0 or lon != 0.0)
        p_lon, p_lat, p_speed = self._get_raw_calc(jd, planet, topocentric=has_observer, lat=lat, lon=lon, elev=elev)

        # Sun position for combustion
        if planet == Planet.SUN:
            sun_lon = p_lon
        else:
            sun_lon, _, _ = self._get_raw_calc(jd, Planet.SUN, topocentric=False)

        motion = self.get_motion_state(p_speed)
        is_combust = self.is_planet_combust(planet, p_lon, sun_lon, is_retro=(motion == MotionState.RETROGRADE))

        # Sign
        sign_idx = int(p_lon // 30.0) % 12
        sign_name = ZODIAC_SIGNS[sign_idx]
        deg_in_sign = p_lon % 30.0

        # Nakshatra
        nak_span = 360.0 / 27.0
        nak_idx = int(p_lon // nak_span) % 27
        nak_name = NAKSHATRAS[nak_idx]

        # Pada (1 to 4)
        pada_span = 360.0 / 108.0
        pada = int((p_lon // pada_span) % 4) + 1

        alt_deg, is_above = None, None
        if has_observer:
            # Topocentric horizon altitude
            geopos = (lon, lat, elev)
            # Equatorial coords for altaz calculation
            eq_flags = swe.FLG_SWIEPH | swe.FLG_EQUATORIAL
            p_code = SWISSEPH_PLANETS.get(planet.value, swe.SUN)
            if planet != Planet.KETU and isinstance(p_code, int):
                try:
                    res_eq, _ = swe.calc_ut(jd, p_code, eq_flags)
                    ra, dec = res_eq[0], res_eq[1]
                    # Convert to Alt/Az using EQU2HOR
                    altaz = swe.azalt(jd, swe.EQU2HOR, geopos, 1013.25, 10.0, (ra, dec, 1.0))
                    alt_deg = round(altaz[1], 4)
                    is_above = (alt_deg > 0.0)
                except Exception:
                    pass

        return PlanetPosition(
            planet=planet,
            longitude=p_lon,
            latitude=p_lat,
            speed=p_speed,
            motion=motion,
            sign=sign_name,
            sign_index=sign_idx + 1,
            degree_in_sign=deg_in_sign,
            nakshatra=nak_name,
            nakshatra_index=nak_idx + 1,
            pada=pada,
            dms=format_dms_str(p_lon),
            is_combust=is_combust,
            altitude_deg=alt_deg,
            is_above_horizon=is_above
        )

    def get_all_positions(self, dt: datetime, lat: float = 0.0, lon: float = 0.0, elev: float = 0.0) -> dict[Planet, PlanetPosition]:
        result = {}
        for p in (Planet.SUN, Planet.MOON, Planet.MARS, Planet.MERCURY, Planet.JUPITER, Planet.VENUS, Planet.SATURN, Planet.RAHU, Planet.KETU):
            result[p] = self.get_planet_position(p, dt, lat=lat, lon=lon, elev=elev)
        return result
