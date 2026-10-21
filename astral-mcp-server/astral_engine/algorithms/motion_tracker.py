"""
Tracks planetary motion changes and retrograde intervals via bisection root-finding.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List
from ..core.models import Planet, Period, DateRange
from ..core.algebra import IntervalSet
from ..calcs.sidereal_calc import SiderealCalc
from .bisection import bisection_search

class MotionTracker:
    def __init__(self, calc: SiderealCalc):
        self.calc = calc

    def get_speed(self, planet: Planet, dt: datetime) -> float:
        pos = self.calc.get_planet_position(planet, dt)
        return pos.speed

    def find_retrograde_periods(self, planet: Planet, drange: DateRange, step_days: int = 1) -> IntervalSet:
        """Finds all retrograde periods using bisection search on speed zero-crossings."""
        if planet in (Planet.SUN, Planet.MOON, Planet.RAHU, Planet.KETU):
            # Sun and Moon never go retrograde; Rahu/Ketu are always retrograde in mean motion
            if planet in (Planet.RAHU, Planet.KETU):
                return IntervalSet([Period(start=drange.start, end=drange.end, event_type="retrograde", planet=planet)])
            return IntervalSet([])

        intervals: List[Period] = []
        prev_dt = drange.start
        prev_speed = self.get_speed(planet, prev_dt)
        in_retro = (prev_speed < 0)
        curr_start = prev_dt if in_retro else None

        for curr_dt in drange.days(step=step_days):
            curr_speed = self.get_speed(planet, curr_dt)

            if not in_retro and prev_speed > 0 and curr_speed <= 0:
                # Direct -> Retrograde (Stationary Retrograde)
                in_retro = True
                fn = lambda t: self.get_speed(planet, t)
                exact_start = bisection_search(fn, prev_dt, curr_dt, tol_seconds=5.0)
                curr_start = exact_start

            elif in_retro and prev_speed < 0 and curr_speed >= 0:
                # Retrograde -> Direct (Stationary Direct)
                in_retro = False
                fn = lambda t: self.get_speed(planet, t)
                exact_end = bisection_search(fn, prev_dt, curr_dt, tol_seconds=5.0)
                if curr_start:
                    intervals.append(Period(
                        start=curr_start,
                        end=exact_end,
                        event_type="retrograde",
                        planet=planet,
                        details={"planet": planet.value, "state": "Retrograde"}
                    ))
                curr_start = None

            prev_dt = curr_dt
            prev_speed = curr_speed

        if in_retro and curr_start:
            intervals.append(Period(
                start=curr_start,
                end=drange.end,
                event_type="retrograde",
                planet=planet,
                details={"planet": planet.value, "state": "Retrograde", "partial": True}
            ))

        return IntervalSet(intervals)
