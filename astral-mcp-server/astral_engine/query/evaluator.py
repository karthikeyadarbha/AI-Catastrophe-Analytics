"""
Declarative query evaluator for composite multi-condition interval algebra.
"""
from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..core.models import Planet, DateRange, Period, MotionState
from ..core.algebra import IntervalSet
from ..calcs.sidereal_calc import SiderealCalc
from ..calcs.chara_karaka import compute_chara_karakas
from ..algorithms.motion_tracker import MotionTracker
from ..algorithms.combustion_tracker import CombustionTracker
from ..algorithms.transit_tracker import TransitTracker
from ..algorithms.aspect_engine import AspectEngine
from ..utils import parse_datetime

class QueryEvaluator:
    def __init__(self, calc: SiderealCalc):
        self.calc = calc
        self.motion_tracker = MotionTracker(calc)
        self.combustion_tracker = CombustionTracker(calc)
        self.transit_tracker = TransitTracker(calc)
        self.aspect_engine = AspectEngine(calc)

    def _eval_single_filter(self, f: Dict[str, Any], drange: DateRange) -> IntervalSet:
        p_str = f.get("planet")
        planet = Planet.from_string(p_str) if p_str else None
        cond = f.get("condition", "").lower()
        val = f.get("value")

        if cond == "motion":
            target_state = str(val).lower()
            if "retro" in target_state:
                return self.motion_tracker.find_retrograde_periods(planet, drange)
            elif "direct" in target_state:
                retro_set = self.motion_tracker.find_retrograde_periods(planet, drange)
                return retro_set.invert(drange)

        elif cond in ("sign", "rasi"):
            if isinstance(val, list):
                # Union of signs
                res = IntervalSet([])
                for s in val:
                    res = res.union(self.transit_tracker.find_sign_intervals(planet, s, drange))
                return res
            else:
                return self.transit_tracker.find_sign_intervals(planet, str(val), drange)

        elif cond in ("combust", "combustion"):
            is_combust = bool(val)
            combust_set = self.combustion_tracker.find_combustion_periods(planet, drange)
            return combust_set if is_combust else combust_set.invert(drange)

        elif cond in ("chara_karaka", "karaka"):
            target_karaka = str(val).upper()
            # Scan daily to find when planet holds this karaka
            periods: List[Period] = []
            curr_start = None
            for curr_dt in drange.days(step=1):
                all_pos = self.calc.get_all_positions(curr_dt)
                karaka_map = compute_chara_karakas(all_pos)
                holds = (karaka_map.get(target_karaka) == planet.value)

                if holds and curr_start is None:
                    curr_start = curr_dt
                elif not holds and curr_start is not None:
                    periods.append(Period(
                        start=curr_start,
                        end=curr_dt,
                        event_type=f"{planet.value} as {target_karaka}",
                        planet=planet,
                        details={"planet": planet.value, "karaka": target_karaka}
                    ))
                    curr_start = None

            if curr_start is not None:
                periods.append(Period(
                    start=curr_start,
                    end=drange.end,
                    event_type=f"{planet.value} as {target_karaka}",
                    planet=planet,
                    details={"planet": planet.value, "karaka": target_karaka, "partial": True}
                ))
            return IntervalSet(periods)

        elif cond == "stellium":
            min_p = int(val) if val else 3
            return self.aspect_engine.find_stelliums(drange, min_planets=min_p)

        elif cond in ("aspect", "alignment", "syzygy"):
            b1 = f.get("planet") or f.get("body1") or "Moon"
            b2 = f.get("target_planet") or f.get("body2") or "Sun"
            target_aspect = f.get("aspect")
            orb = float(f.get("orb", 10.0 if cond == "syzygy" else 6.0))
            return self.aspect_engine.find_aspect_intervals(
                body1=b1,
                body2=b2,
                drange=drange,
                target_aspect=target_aspect,
                orb_deg=orb,
            )

        # Fallback empty
        return IntervalSet([])

    def execute_query(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        filters: List[Dict[str, Any]],
        operator: str = "AND",
        min_duration_hours: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Compiles and evaluates composite interval algebra query."""
        drange = DateRange(parse_datetime(start_date), parse_datetime(end_date))
        if not filters:
            return []

        op = operator.strip().upper()
        evaluated_sets: List[IntervalSet] = []

        for f in filters:
            s = self._eval_single_filter(f, drange)
            evaluated_sets.append(s)

        if op == "AND":
            final_set = evaluated_sets[0]
            for s in evaluated_sets[1:]:
                final_set = final_set.intersect(s)
        elif op == "OR":
            final_set = evaluated_sets[0]
            for s in evaluated_sets[1:]:
                final_set = final_set.union(s)
        else:
            raise ValueError(f"Unknown operator '{operator}'. Use 'AND' or 'OR'.")

        if min_duration_hours > 0:
            final_set = final_set.filter_min_duration(min_duration_hours)

        return final_set.to_dicts()
