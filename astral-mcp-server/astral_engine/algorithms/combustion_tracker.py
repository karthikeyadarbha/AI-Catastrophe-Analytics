"""
Tracks solar combustion intervals via bisection root-finding on angular separation.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List
from ..core.models import Planet, Period, DateRange, MotionState
from ..core.algebra import IntervalSet
from ..calcs.sidereal_calc import SiderealCalc
from ..constants import COMBUSTION_LIMITS
from ..utils import minimal_angular_diff
from .bisection import bisection_search

class CombustionTracker:
    def __init__(self, calc: SiderealCalc):
        self.calc = calc

    def _get_threshold(self, planet: Planet, dt: datetime) -> float:
        limits = COMBUSTION_LIMITS.get(planet.value)
        if not limits:
            return 0.0
        pos = self.calc.get_planet_position(planet, dt)
        is_retro = (pos.motion == MotionState.RETROGRADE)
        return limits["retro"] if is_retro else limits["direct"]

    def _separation(self, planet: Planet, dt: datetime) -> float:
        p_pos = self.calc.get_planet_position(planet, dt)
        s_pos = self.calc.get_planet_position(Planet.SUN, dt)
        return minimal_angular_diff(p_pos.longitude, s_pos.longitude)

    def find_combustion_periods(self, planet: Planet, drange: DateRange, step_days: int = 1) -> IntervalSet:
        if planet in (Planet.SUN, Planet.RAHU, Planet.KETU):
            return IntervalSet([])

        intervals: List[Period] = []
        prev_dt = drange.start
        prev_sep = self._separation(planet, prev_dt)
        prev_thresh = self._get_threshold(planet, prev_dt)
        in_combust = (prev_sep < prev_thresh)
        curr_start = prev_dt if in_combust else None

        for curr_dt in drange.days(step=step_days):
            curr_sep = self._separation(planet, curr_dt)
            curr_thresh = self._get_threshold(planet, curr_dt)

            if not in_combust and prev_sep >= prev_thresh and curr_sep < curr_thresh:
                # Entering combustion
                in_combust = True
                fn = lambda t: self._separation(planet, t) - self._get_threshold(planet, t)
                exact_start = bisection_search(fn, prev_dt, curr_dt, tol_seconds=5.0)
                curr_start = exact_start

            elif in_combust and prev_sep < prev_thresh and curr_sep >= curr_thresh:
                # Exiting combustion
                in_combust = False
                fn = lambda t: self._separation(planet, t) - self._get_threshold(planet, t)
                exact_end = bisection_search(fn, prev_dt, curr_dt, tol_seconds=5.0)
                if curr_start:
                    intervals.append(Period(
                        start=curr_start,
                        end=exact_end,
                        event_type="combustion",
                        planet=planet,
                        details={"planet": planet.value, "combust": True}
                    ))
                curr_start = None

            prev_dt = curr_dt
            prev_sep = curr_sep
            prev_thresh = curr_thresh

        if in_combust and curr_start:
            intervals.append(Period(
                start=curr_start,
                end=drange.end,
                event_type="combustion",
                planet=planet,
                details={"planet": planet.value, "combust": True, "partial": True}
            ))

        return IntervalSet(intervals)
