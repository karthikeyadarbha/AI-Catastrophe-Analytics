import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indic_engine.compute import drk, positions as P
from indic_engine.rules import get_rule, search_rules
def test_ahargana_mesa_2002():
    assert abs(drk.mesa_ahargana(1924) - 1863913.0) < 1.0
def test_tables_shape_and_peaks():
    for t in drk.MANDA.values(): assert len(t) == 15 and all(b > a for a, b in zip(t, t[1:])) or t[-1] >= t[-2]
    for b, (m, k) in drk.SIGHRA.items(): assert len(m) == len(k) == 15 and m[-1] == k[-1]
    assert drk.MANDA["Mars"][-1] == 707 and drk.MANDA["Sun"][-1] == 129 and drk.MANDA["Moon"][-1] == 301
def test_interp_at_nodes():
    assert abs(drk._interp(drk.MANDA["Mars"], 12.0) - 133) < 1e-9 and abs(drk._interp(drk.SIGHRA["Venus"][1], 90.0) - 2150) < 1e-9
def test_sun_agrees_with_ts():
    j = 2451545.0; d = drk.planet_positions(j)["true"]["Sun"]; t = P.planet_positions(j, "TS")["true"]["Sun"]
    assert abs((d - t + 180) % 360 - 180) < 0.15
def test_zero_point_is_parahita_522():
    import math
    def tm(jd): T = (jd - 2451545.0) / 36525; return (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360
    f = lambda jd: ((tm(jd) - drk.mean_longitudes(jd)["Sun"] + 180) % 360) - 180
    lo, hi = 2451545.0 - 2000 * 365.25, 2451545.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if f(mid) > 0: hi = mid
        else: lo = mid
    assert abs(2000 + (mid - 2451545.0) / 365.25 - 522.2) < 1.0
def test_registry():
    assert len(search_rules(text="Drgganita of Paramesvara (ed. K.V. Sarma, Hoshiarpur 1963; Saka 1353 = 1431 CE)")) == 17 and get_rule("DG.mean")["status"] == "implemented"
