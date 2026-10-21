import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indic_engine.compute import ss
from indic_engine.rules import get_rule, search_rules
D = 1834977                                  # book's Vikrama 1978 example (Examples 1-2, Spashtadhikara commentary)
Y = 1_955_885_023
def dms(s, d, m, sec): return s * 30 + d + m / 60 + sec / 3600
def close(a, b, tol): return abs((a - b + 180) % 360 - 180) < tol
def test_rsine_table():
    assert ss.RS[-1] == 3438 and ss.RS[17] == 3084 and ss.RS[18] == 3177 and len(ss.RSINE_DIFF) == 24
    assert abs(ss.rsin(90) - 3438) < 1e-9 and abs(ss.rsin(30) - 1719) < 1e-9 and ss.rsin(210) < 0
def test_rsine_book_interpolation():            # book p.129: Rsin 66 deg = 3140'
    assert abs(ss.rsin(66) - 3140) < 1.5
    assert abs(ss.rarcsin(3140) - 66) < 0.01
def test_means_book_example():
    m = ss.mean_longitudes(0, D=D)
    assert close(m["Jupiter"], dms(6, 19, 52, 37), 2 / 3600)
    assert close(m["Mercury"], dms(2, 9, 29, 17), 3 / 3600)      # Mercury sighra mean
    assert close(m["Sun"], dms(9, 8, 12, 6), 5 / 3600)
def test_apsides_book_example():
    for b, tgt in (("Sun", dms(2, 17, 17, 31.17)), ("Mercury", dms(7, 10, 28, 26.54)), ("Jupiter", dms(5, 21, 22, 36.21))):
        v = ((ss.KALPA_REV[b] * Y / 4.32e9) % 1) * 360
        assert close(v, tgt, 0.2 / 3600), b
def test_kali_start_apsides():
    assert close(ss.apsides(0, D=0.0)[0]["Sun"], 77.12, 0.05)       # about 77.1 deg; today's value ~77.3
    m = ss.mean_longitudes(0, D=0.0)
    assert abs(m["MoonApogee"] - 90) < 1e-9 and abs(m["MoonNode"] - 180) < 1e-9 and all(m[k] == 0 for k in ("Sun", "Moon", "Mars", "Jupiter"))
def test_epicycle_values():
    assert abs(ss.rarcsin(14 * 3438 / 360) * 3600 - (2 * 3600 + 13 * 60 + 41)) < 3              # commentator quotes 2d13'41'' for the Sun's full 14 deg epicycle (p.132)
    assert abs(ss.manda_phala("Sun", 90) * 60 - 130.5) < 1.0                                       # at anomaly 90 the odd-pada circumference 13d40' applies
    assert abs(ss.manda_circumference("Sun", 0) - 14) < 1e-9 and abs(ss.manda_circumference("Sun", 90) - (14 - 20 / 60)) < 1e-9
    assert ss.sighra_circumference("Jupiter", 90) == 72 and ss.sighra_circumference("Jupiter", 0) == 70
def test_true_sun_and_jupiter_book_example():
    m = ss.mean_longitudes(0, D=D); ap = {b: ((k * Y / 4.32e9) % 1) * 360 for b, k in ss.KALPA_REV.items()}
    assert close(ss.true_sun(m, ap, "book"), dms(9, 8, 59, 6), 40 / 3600)
    assert close(ss.true_planet("Jupiter", m, ap, "book"), dms(6, 27, 53, 37), 30 / 3600)
    assert close(ss.true_planet("Jupiter", m, ap), dms(6, 27, 53, 37), 4 / 60)                          # literal inverse-Rsine rule: within 4 arc-minutes
def test_zero_point_516():
    def tm(jd): T = (jd - 2451545.0) / 36525; return (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360
    f = lambda jd: ((tm(jd) - ss.mean_longitudes(jd)["Sun"] + 180) % 360) - 180
    lo, hi = 2451545.0 - 3000 * 365.25, 2451545.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if f(mid) > 0: hi = mid
        else: lo = mid
    assert abs(2000 + (mid - 2451545.0) / 365.25 - 516.4) < 1.0
def test_retrograde_2022_23():
    assert ss.is_retrograde("Mars", 2459915.5) and not ss.is_retrograde("Mars", 2460036.5)       # 1 Dec 2022 retrograde, 1 Apr 2023 direct
    assert ss.is_retrograde("Jupiter", 2459853.5)                                                  # 1 Oct 2022
    assert ss.is_retrograde("Saturn", 2460188.5)                                                   # 1 Sep 2023
def test_agrees_with_ts_roughly():
    from indic_engine.compute import positions as P
    j = 2451545.0; s = ss.planet_positions(j)["true"]; t = P.planet_positions(j, "TS")["true"]
    for b in ("Sun", "Moon"): assert abs((s[b] - t[b] + 180) % 360 - 180) < 2.5, b
def test_registry():
    assert len(search_rules(text="Suryasiddhanta (Hindi Vijnana-bhashya edition, Pt. Mahavir Prasad Srivastava; PDF 838 pp.)")) == 16 and get_rule("SS.concordance")["status"] == "implemented"
    assert get_rule("SS.apsides")["status"] == "implemented" and get_rule("SS.daily.motion")["status"] == "registered-only"
def test_sanskrit_edition_concordance():
    # Pandey / Chaukhamba Sanskrit edition: Rsine table 2.23-27 (arc-complement values printed as utkrama-jya) and kshepaka table of apsides at the end of Krta (v.58)
    V = [7,29,66,117,182,261,354,460,579,710,853,1007,1171,1345,1528,1719,1918,2123,2333,2548,2767,2989,3213,3438]
    assert [3438 - ss.RS[24 - k] for k in range(1, 25)] == V
    Yk = 1_953_720_000
    want = {"Sun": (0,7,28,12), "Mars": (3,3,14,24), "Mercury": (5,4,4,48), "Jupiter": (0,9,0,0), "Venus": (11,13,21,0)}
    for b, (s_, d, m, sec) in want.items():
        assert close(((ss.KALPA_REV[b] * Yk / 4.32e9) % 1) * 360, dms(s_, d, m, sec), 2 / 3600), b
    nodes = {"Mars": (9,11,20,12), "Mercury": (8,11,16,48), "Jupiter": (8,8,56,24), "Venus": (4,17,25,48), "Saturn": (4,20,13,12)}
    for b, (s_, d, m, sec) in nodes.items():
        assert close(-((ss.NODE_KALPA[b] * Yk / 4.32e9) % 1) * 360, dms(s_, d, m, sec), 15 / 3600), b
    assert ss.SIGHRA_EVEN["Saturn"] == 39
