"""
Continuous bisection root-finding algorithm for precise temporal boundary detection.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Callable, Any

def bisection_search(
    func: Callable[[datetime], float],
    t1: datetime,
    t2: datetime,
    tol_seconds: float = 1.0,
    max_iter: int = 40,
) -> datetime:
    """
    Finds root t* in [t1, t2] where func(t*) crosses zero.
    Assumes func(t1) and func(t2) have opposite signs (or bracket the target).
    """
    f1 = func(t1)
    f2 = func(t2)

    if abs(f1) < 1e-9:
        return t1
    if abs(f2) < 1e-9:
        return t2

    for _ in range(max_iter):
        delta = (t2 - t1).total_seconds()
        if abs(delta) <= tol_seconds:
            break
        
        mid = t1 + timedelta(seconds=delta / 2.0)
        f_mid = func(mid)

        if abs(f_mid) < 1e-9:
            return mid

        if (f1 > 0 and f_mid > 0) or (f1 < 0 and f_mid < 0):
            t1 = mid
            f1 = f_mid
        else:
            t2 = mid
            f2 = f_mid

    return t1 + (t2 - t1) / 2.0

def binary_search_discrete(
    func: Callable[[datetime], Any],
    t1: datetime,
    t2: datetime,
    tol_seconds: float = 1.0,
    max_iter: int = 40,
) -> datetime:
    """
    Binary search between t1 and t2 to find transition point where discrete func(t) changes value.
    Returns the first timestamp of the new value.
    """
    val1 = func(t1)
    for _ in range(max_iter):
        delta = (t2 - t1).total_seconds()
        if abs(delta) <= tol_seconds:
            break
        mid = t1 + timedelta(seconds=delta / 2.0)
        if func(mid) == val1:
            t1 = mid
        else:
            t2 = mid
    return t2
