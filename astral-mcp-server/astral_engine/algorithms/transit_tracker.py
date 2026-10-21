"""
Tracks sign (Rāśi) and Nakṣatra boundary transits via binary search.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Literal
from ..core.models import Planet, Period, DateRange, TransitionEvent
from ..core.algebra import IntervalSet
from ..calcs.sidereal_calc import SiderealCalc
from .bisection import binary_search_discrete

class TransitTracker:
    def __init__(self, calc: SiderealCalc):
        self.calc = calc

    def find_sign_transits(self, planet: Planet, drange: DateRange, step_days: int = 1) -> List[TransitionEvent]:
        """Finds all exact moments where a planet enters a new zodiac sign."""
        events: List[TransitionEvent] = []
        prev_dt = drange.start
        prev_pos = self.calc.get_planet_position(planet, prev_dt)
        prev_sign = prev_pos.sign

        for curr_dt in drange.days(step=step_days):
            curr_pos = self.calc.get_planet_position(planet, curr_dt)
            curr_sign = curr_pos.sign

            if curr_sign != prev_sign:
                # Sign change occurred in [prev_dt, curr_dt]
                fn = lambda t: self.calc.get_planet_position(planet, t).sign
                exact_t = binary_search_discrete(fn, prev_dt, curr_dt, tol_seconds=10.0)
                new_pos = self.calc.get_planet_position(planet, exact_t)
                
                events.append(TransitionEvent(
                    timestamp=exact_t,
                    event_type="rasi_transit",
                    planet=planet,
                    description=f"{planet.value} entered {new_pos.sign}",
                    old_value=prev_sign,
                    new_value=new_pos.sign
                ))
                prev_sign = new_pos.sign

            prev_dt = curr_dt

        return events

    def find_nakshatra_transits(self, planet: Planet, drange: DateRange, step_days: int = 1) -> List[TransitionEvent]:
        """Finds all exact moments where a planet enters a new Nakṣatra."""
        events: List[TransitionEvent] = []
        prev_dt = drange.start
        prev_pos = self.calc.get_planet_position(planet, prev_dt)
        prev_nak = prev_pos.nakshatra

        for curr_dt in drange.days(step=step_days):
            curr_pos = self.calc.get_planet_position(planet, curr_dt)
            curr_nak = curr_pos.nakshatra

            if curr_nak != prev_nak:
                fn = lambda t: self.calc.get_planet_position(planet, t).nakshatra
                exact_t = binary_search_discrete(fn, prev_dt, curr_dt, tol_seconds=10.0)
                new_pos = self.calc.get_planet_position(planet, exact_t)

                events.append(TransitionEvent(
                    timestamp=exact_t,
                    event_type="nakshatra_transit",
                    planet=planet,
                    description=f"{planet.value} entered {new_pos.nakshatra} (Pada {new_pos.pada})",
                    old_value=prev_nak,
                    new_value=new_pos.nakshatra
                ))
                prev_nak = new_pos.nakshatra

            prev_dt = curr_dt

        return events

    def find_sign_intervals(self, planet: Planet, sign_name: str, drange: DateRange, step_days: int = 1) -> IntervalSet:
        """Returns intervals where the planet is residing in sign_name."""
        periods: List[Period] = []
        prev_dt = drange.start
        in_sign = (self.calc.get_planet_position(planet, prev_dt).sign.lower() == sign_name.lower())
        curr_start = prev_dt if in_sign else None

        for curr_dt in drange.days(step=step_days):
            is_now_in = (self.calc.get_planet_position(planet, curr_dt).sign.lower() == sign_name.lower())

            if not in_sign and is_now_in:
                # Entering
                in_sign = True
                fn = lambda t: self.calc.get_planet_position(planet, t).sign.lower() == sign_name.lower()
                exact_start = binary_search_discrete(lambda t: not fn(t), prev_dt, curr_dt, tol_seconds=10.0)
                curr_start = exact_start

            elif in_sign and not is_now_in:
                # Exiting
                in_sign = False
                fn = lambda t: self.calc.get_planet_position(planet, t).sign.lower() == sign_name.lower()
                exact_end = binary_search_discrete(fn, prev_dt, curr_dt, tol_seconds=10.0)
                if curr_start:
                    periods.append(Period(
                        start=curr_start,
                        end=exact_end,
                        event_type="in_sign",
                        planet=planet,
                        details={"planet": planet.value, "sign": sign_name}
                    ))
                curr_start = None

            prev_dt = curr_dt

        if in_sign and curr_start:
            periods.append(Period(
                start=curr_start,
                end=drange.end,
                event_type="in_sign",
                planet=planet,
                details={"planet": planet.value, "sign": sign_name, "partial": True}
            ))

        return IntervalSet(periods)
