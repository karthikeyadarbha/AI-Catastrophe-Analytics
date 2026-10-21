import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indic_engine.rules import RULES, get_rule, search_rules, list_sources
from indic_engine.compute import series, planet_positions

def test_registry_integrity():
    ids = [r["id"] for r in RULES]; assert len(ids) == len(set(ids)) >= 30
    for r in RULES:
        assert r["status"] in ("implemented", "registered-only") and r["reference"] and r["text"]
        if r["sanskrit_verified"]: assert r["sanskrit"]

def test_flagged_discrepancy():
    assert "117" in get_rule("AB.jya")["notes"]

def test_series_vs_exact():
    R = series.R_EXACT
    assert abs(series.madhava_sine(1500) - R*math.sin(1500/R)) < 1e-9
    assert abs(series.madhava_cosine(1500) - R*math.cos(1500/R)) < 1e-9
    assert abs(series.madhava_arctan(1, 3, terms=60) - R*math.atan(1/3)) < 1e-6
    x = 0.9; assert abs(series.madhava_arctan_corrected(x, 6)-math.atan(x)) < abs(sum((-1)**k*x**(2*k+1)/(2*k+1) for k in range(6))-math.atan(x))

def test_param_sets_both_run():
    for p in ("TS", "AB"):
        r = planet_positions(2451545.0, p); assert set(r["true"]) >= {"Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"}
