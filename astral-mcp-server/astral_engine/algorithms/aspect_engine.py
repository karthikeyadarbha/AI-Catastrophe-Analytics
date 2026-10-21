"""
Aspects, Pairwise Angular Relationships, Stelliums, and Alignment analysis.
Universal treatment across all celestial bodies without planetary bias.
"""
from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from ..core.models import Planet, Period, DateRange
from ..core.algebra import IntervalSet
from ..calcs.sidereal_calc import SiderealCalc
from ..constants import ZODIAC_SIGNS
from ..utils import minimal_angular_diff

class AspectEngine:
    """
    Universal engine for computing geometric angular relationships,
    aspects, multi-body alignments, and stelliums across all planets.
    """
    def __init__(self, calc: SiderealCalc):
        self.calc = calc

    def get_angular_relationship(
        self,
        body1: Planet | str,
        body2: Planet | str = Planet.SUN,
        dt: datetime = None,
        orb_deg: float = 6.0,
    ) -> dict:
        """
        Computes the exact geometric angular relationship between ANY two celestial bodies:
        - Angular separation (0° - 180°)
        - Longitudinal difference (0° - 360°, body1 relative to body2)
        - Major aspects (Conjunction 0°, Sextile 60°, Square 90°, Trine 120°, Opposition 180°)
        - Collinear alignment / Syzygy flag (separation <= orb or >= 180 - orb)
        - Illumination percentage (general phase formula when body2 or body1 is Sun)
        """
        p1 = Planet.from_string(body1) if isinstance(body1, str) else body1
        p2 = Planet.from_string(body2) if isinstance(body2, str) else body2

        pos1 = self.calc.get_planet_position(p1, dt)
        pos2 = self.calc.get_planet_position(p2, dt)

        # Angular difference (0..180)
        angular_sep = minimal_angular_diff(pos1.longitude, pos2.longitude)
        # Directional difference (0..360)
        long_diff = (pos1.longitude - pos2.longitude) % 360.0

        # Aspects detection with tolerance orb
        aspects = {
            "Conjunction": 0.0,
            "Sextile": 60.0,
            "Square": 90.0,
            "Trine": 120.0,
            "Opposition": 180.0,
        }
        detected_aspect = None
        min_delta = orb_deg
        for name, exact_angle in aspects.items():
            delta = abs(angular_sep - exact_angle)
            if delta <= orb_deg and delta < min_delta:
                min_delta = delta
                detected_aspect = name

        # Collinear alignment (Conjunction or Opposition within orb)
        is_aligned = (angular_sep <= orb_deg or angular_sep >= (180.0 - orb_deg))

        # Illumination percentage (valid phase formula relative to illumination source Sun)
        has_sun = (p1 == Planet.SUN or p2 == Planet.SUN)
        illumination_pct = round(0.5 * (1.0 + math.cos(math.radians(angular_sep))) * 100.0, 2) if has_sun else None

        return {
            "datetime": dt.isoformat(),
            "body1": p1.value,
            "body1_longitude": round(pos1.longitude, 4),
            "body1_sign": pos1.sign,
            "body2": p2.value,
            "body2_longitude": round(pos2.longitude, 4),
            "body2_sign": pos2.sign,
            "angular_separation_deg": round(angular_sep, 4),
            "longitudinal_difference_deg": round(long_diff, 4),
            "aspect": detected_aspect,
            "is_aligned": is_aligned,
            "illumination_pct": illumination_pct,
        }

    def find_aspect_intervals(
        self,
        body1: Planet | str,
        body2: Planet | str,
        drange: DateRange,
        target_aspect: Optional[str] = None,
        orb_deg: float = 6.0,
        step_days: int = 1,
    ) -> IntervalSet:
        """
        Scans a date range and returns intervals where two bodies form a specific aspect
        or general collinear alignment.
        """
        p1 = Planet.from_string(body1) if isinstance(body1, str) else body1
        p2 = Planet.from_string(body2) if isinstance(body2, str) else body2

        periods: List[Period] = []
        curr_start: Optional[datetime] = None
        last_details: Optional[dict] = None

        for curr_dt in drange.days(step=step_days):
            rel = self.get_angular_relationship(p1, p2, curr_dt, orb_deg=orb_deg)
            if target_aspect:
                matched = (rel["aspect"] == target_aspect)
            else:
                matched = rel["is_aligned"]

            if matched and curr_start is None:
                curr_start = curr_dt
                last_details = rel
            elif not matched and curr_start is not None:
                periods.append(Period(
                    start=curr_start,
                    end=curr_dt,
                    event_type=f"{p1.value}-{p2.value} {target_aspect or 'Alignment'}",
                    details=last_details or rel,
                ))
                curr_start = None
                last_details = None

        if curr_start is not None:
            periods.append(Period(
                start=curr_start,
                end=drange.end,
                event_type=f"{p1.value}-{p2.value} {target_aspect or 'Alignment'}",
                details={"partial": True, **(last_details or {})},
            ))

        return IntervalSet(periods)

    def find_stelliums(self, drange: DateRange, min_planets: int = 3, step_days: int = 1) -> IntervalSet:
        """Finds intervals where min_planets or more occupy the exact same zodiac sign."""
        tracked_planets = [
            Planet.SUN, Planet.MOON, Planet.MARS, Planet.MERCURY,
            Planet.JUPITER, Planet.VENUS, Planet.SATURN, Planet.RAHU
        ]
        periods: List[Period] = []
        active_stelliums: Dict[str, tuple[datetime, list[str]]] = {}

        for curr_dt in drange.days(step=step_days):
            # Group planets by sign
            sign_occupants: Dict[str, list[str]] = {s: [] for s in ZODIAC_SIGNS}
            for p in tracked_planets:
                pos = self.calc.get_planet_position(p, curr_dt)
                sign_occupants[pos.sign].append(p.value)

            for sign, occupants in sign_occupants.items():
                count = len(occupants)
                if count >= min_planets and sign not in active_stelliums:
                    active_stelliums[sign] = (curr_dt, occupants)
                elif count < min_planets and sign in active_stelliums:
                    start_t, initial_planets = active_stelliums.pop(sign)
                    periods.append(Period(
                        start=start_t,
                        end=curr_dt,
                        event_type="stellium",
                        details={
                            "sign": sign,
                            "min_planets": min_planets,
                            "planets": initial_planets,
                            "count": len(initial_planets)
                        }
                    ))

        # Close remaining
        for sign, (start_t, initial_planets) in active_stelliums.items():
            periods.append(Period(
                start=start_t,
                end=drange.end,
                event_type="stellium",
                details={
                    "sign": sign,
                    "min_planets": min_planets,
                    "planets": initial_planets,
                    "count": len(initial_planets),
                    "partial": True
                }
            ))

        return IntervalSet(periods)
