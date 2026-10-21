import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indic_engine.compute import ayanamsa as A, panchanga as Pn, conjunction as C, occultation as O, visibility as V
def dms(x): return x
def test_zero_years_match_published():
    exp = {"Lahiri": 285.7, "Raman": 390.0, "KP": 292.7, "ChandraHari": 233.1, "Dulakara": 243.9}
    for m, y in exp.items(): assert abs(A.zero_year(m) - y) < 1.0, m
def test_overbeck_values():
    j = A.jd_from_ymd(2006, 10, 27, 23.768)
    for m, v in {"Lahiri": 23+57/60+10/3600, "Raman": 22+30/60+24/3600, "KP": 23+51/60+22/3600}.items():
        assert abs(A.ayanamsa(j, m) - v) < 3/3600, m
def test_dulakara_table_2000():
    assert abs(A.ayanamsa(A.jd_from_ymd(2000, 1, 1), "Dulakara") - (24+25/60+37/3600)) < 10/3600
def test_treatise_zero_at_kali_3600():
    j = A.KALI_JD + 3600*365.2425
    assert abs(A.treatise_aryabhata_school(j)) < 1e-6 and abs(A.treatise_surya_nilakantha(A.KALI_JD + 3600*365.258756)) < 1e-6
def test_aryabhata_school_2000():
    assert abs(A.ayanamsa(A.jd_from_ymd(2000, 1, 1), "AryabhataSchool") - 23.19) < 0.05
def test_panchanga_basics():
    assert Pn.tithi(0, 12.0)["number"] == 2 and Pn.nakshatra(49.5)["name"] == "Rohini" and Pn.karana(0, 6.5)["name"] == "Bava"
def test_conjunction_and_bheda():
    assert abs(C.time_of_conjunction(10, 12, 1, 0.5) - 4.0) < 1e-9
    assert C.yuddha(1.0, "Mars", "Jupiter")["type"] == "bheda" and C.yuddha(30.0, "Mars", "Jupiter")["type"].startswith("beyond")
def test_rohini_and_visibility():
    assert O.rohini_sakata_bheda(47.2, -2.3) and not O.rohini_sakata_bheda(47.2, -1.0) and O.rohini_sakata_bheda(47.0, -2.2, rule="ganesa")
    assert V.visible("Mars", 100, 80)["visible"] and not V.visible("Mars", 90, 80)["visible"]

def test_chandrahari_primary_anchor():
    assert abs(A.ayanamsa(A.jd_from_ymd(2000,1,1,12), "ChandraHari") - (24+35/60+8.62/3600)) < 2/3600 + 0.0003
    assert abs(A.zero_year("ChandraHari") - 233.1) < 0.2
