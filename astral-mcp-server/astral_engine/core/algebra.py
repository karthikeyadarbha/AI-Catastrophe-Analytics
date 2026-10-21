"""
Allen's Interval Algebra for evaluating complex temporal logic and composite conditions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
from .models import Period, DateRange

@dataclass
class IntervalSet:
    """A collection of non-overlapping, sorted intervals supporting boolean set algebra."""
    intervals: List[Period] = field(default_factory=list)

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        """Sort by start time and merge contiguous or overlapping intervals."""
        if not self.intervals:
            return
        sorted_ints = sorted(self.intervals, key=lambda p: p.start)
        merged = [sorted_ints[0]]
        for curr in sorted_ints[1:]:
            prev = merged[-1]
            if curr.start <= prev.end:
                # Merge
                merged[-1] = Period(
                    start=prev.start,
                    end=max(prev.end, curr.end),
                    event_type=f"{prev.event_type} | {curr.event_type}",
                    planet=prev.planet if prev.planet == curr.planet else None,
                    details={"merged": True}
                )
            else:
                merged.append(curr)
        self.intervals = merged

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self):
        return iter(self.intervals)

    def __getitem__(self, idx):
        return self.intervals[idx]

    def intersect(self, other: IntervalSet) -> IntervalSet:
        """Compute the mathematical intersection of two interval sets (A AND B)."""
        result: List[Period] = []
        i, j = 0, 0
        while i < len(self.intervals) and j < len(other.intervals):
            a = self.intervals[i]
            b = other.intervals[j]
            overlap = a.intersection(b)
            if overlap is not None:
                result.append(overlap)
            if a.end < b.end:
                i += 1
            else:
                j += 1
        return IntervalSet(result)

    def union(self, other: IntervalSet) -> IntervalSet:
        """Compute the mathematical union of two interval sets (A OR B)."""
        combined = list(self.intervals) + list(other.intervals)
        return IntervalSet(combined)

    def invert(self, total_range: DateRange) -> IntervalSet:
        """Compute the complement within total_range (NOT A)."""
        if not self.intervals:
            return IntervalSet([Period(start=total_range.start, end=total_range.end, event_type="unconstrained")])
        
        result: List[Period] = []
        curr = total_range.start
        for p in self.intervals:
            if p.start > curr:
                result.append(Period(start=curr, end=p.start, event_type="complement"))
            curr = max(curr, p.end)
        if curr < total_range.end:
            result.append(Period(start=curr, end=total_range.end, event_type="complement"))
        return IntervalSet(result)

    def filter_min_duration(self, min_hours: float) -> IntervalSet:
        min_sec = min_hours * 3600.0
        filtered = [p for p in self.intervals if p.duration_seconds >= min_sec]
        return IntervalSet(filtered)

    def to_dicts(self) -> List[dict]:
        return [p.to_dict() for p in self.intervals]
