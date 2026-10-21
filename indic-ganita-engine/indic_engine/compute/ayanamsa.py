"""Ayanamsa models. A(t) = tropical minus sidereal longitude, degrees.
Modern family: A(t) = p_A(t) - p_A(t0) [+ nutation], p_A = IAU 2006 general precession in longitude (Capitaine et al. 2003),
 anchored by a published value at a date or a 'zero year' (year of coincidence of the two zodiacs).
Treatise family (Indian Astronomy: A Source Book, ch.14): oscillating ('trepidation') models."""
import math
J2000 = 2451545.0

def p_A(jd):                      # arcsec
    t = (jd - J2000) / 36525.0
    return (((((-0.0000000383*t - 0.000023857)*t + 0.00007964)*t + 1.1054348)*t) + 5028.796195)*t

def nutation_lon(jd):             # arcsec, 4 largest IAU1980 terms (accuracy ~0.1")
    t = (jd - J2000) / 36525.0
    om = math.radians(125.04452 - 1934.136261*t)
    L = math.radians(280.4665 + 36000.7698*t); Lm = math.radians(218.3165 + 481267.8813*t)
    return -17.20*math.sin(om) - 1.32*math.sin(2*L) - 0.23*math.sin(2*Lm) + 0.21*math.sin(2*om)

def jd_from_ymd(y, m, d, h=0.0):  # proleptic Gregorian for y>=1583, Julian before 1582-10-15
    if m <= 2: y -= 1; m += 12
    if (y, m, d) >= (1582, 10, 15) or y > 1582:
        a = y // 100; b = 2 - a + a // 4
    else: b = 0
    return math.floor(365.25*(y+4716)) + math.floor(30.6001*(m+1)) + d + b - 1524.5 + h/24.0

def _dms(d, m, s): return d + m/60 + s/3600

# ---- anchors. value_deg at jd (mean ayanamsa unless noted)
MODERN = {
 "Lahiri":      dict(kind="value", jd=J2000, deg=23.857,  src="LibEphemeris J2000 value; Overbeck table 23d57m10s at 2006-10-27 (consistent)"),
 "Raman":       dict(kind="value", jd=J2000, deg=22.4108, src="LibEphemeris J2000 value; Overbeck 22d30m24s at 2006-10-27 (consistent)"),
 "KP":          dict(kind="value", jd=J2000, deg=23.7602, src="Krishnamurti; LibEphemeris J2000 value; Overbeck 23d51m22s at 2006-10-27 (consistent)"),
 "FaganBradley":dict(kind="value", jd=J2000, deg=24.7403, src="reference only"),
 "ChandraHari": dict(kind="value", jd=J2000, deg=_dms(24,35,8.62), src="K. Chandra Hari, Hindu Zodiac and Ancient Astronomy, Appendix V p.476: Mula fixed at sidereal 240 deg; ayanamsa J2000.0 = 24d35m08.62s (24d35m33.76s at J2000.5); zero precession 1 Feb 233 10:46 (JD 1806192.9486)"),
 "Dulakara":    dict(kind="zero", jd=jd_from_ymd(244,3,20,18.782), src="Indrasena 2011: Spica 29d26'22\" Virgo at vernal equinox 244-03-20 18:46:54 GMT; true precession (nutation included)", nutation=True),
 "Tropical":    dict(kind="zero_all", src="Sayana: ayanamsa = 0 by definition"),
}
ALIASES = {"Indrasena": "Dulakara", "BVRaman": "Raman", "Krishnamurti": "KP", "Chandrahari": "ChandraHari", "Sayana": "Tropical"}

def _modern(model, jd, true):
    s = MODERN[model]
    if s["kind"] == "zero_all": return 0.0
    if s["kind"] == "value": a = s["deg"] + (p_A(jd) - p_A(s["jd"]))/3600.0
    else: a = (p_A(jd) - p_A(s["jd"]))/3600.0
    if s.get("nutation") or true: a += nutation_lon(jd)/3600.0
    return a

def zero_year(model):
    """Year (CE) when the model's ayanamsa is zero (mean), by solving A(t)=0."""
    lo, hi = jd_from_ymd(-500,1,1), J2000
    f = lambda j: _modern(model, j, False)
    for _ in range(80):
        mid = (lo+hi)/2
        if f(lo)*f(mid) <= 0: hi = mid
        else: lo = mid
    return 2000.0 + (lo-J2000)/365.2425

# ---- treatise models (Source Book ch.14). Kali elapsed years from JD (Kali epoch JD 588465.5 as in params)
KALI_JD = 588465.5
def kali_years(jd, year_len=365.2425): return (jd - KALI_JD)/year_len

def _tri(x, period, amp_per_unit_quarter):   # triangle wave (bhuja) used by Suryadeva's rule
    x = x % period; q = period/4
    if x <= q: b = x
    elif x <= 2*q: b = 2*q - x
    elif x <= 3*q: b = -(x - 2*q)
    else: b = -(period - x)
    return b*amp_per_unit_quarter

def treatise_aryabhata_school(jd):
    """Suryadeva (Aryabhatiya commentary) two verses: (K-3600) mod 5808, mod 1452 as bhuja, x2/121 deg. Amplitude 24 deg, zero at Kali 3600."""
    return _tri(kali_years(jd) - 3600.0, 5808.0, 2.0/121.0)

def treatise_surya_nilakantha(jd):
    """Suryasiddhanta 3.9-12 / Nilakantha: 600 revolutions per mahayuga (period 7200 y), A = 3/10 of the bhuja; amplitude 27 deg, zero at Kali 3600.
    Sign: positive after zero."""
    ph = 360.0*kali_years(jd, 365.258756)/7200.0 % 360.0
    if ph <= 90: b = ph
    elif ph <= 180: b = 180 - ph
    elif ph <= 270: b = -(ph - 180)
    else: b = -(360 - ph)
    return -0.3*b

def treatise_deva_amaraja(jd):
    """Deva (Karanaratna 1.3.56) and Amaraja (Khandakhadyaka 3.11): ayana-graha L = 360*K/7380 (K=elapsed Kali years, 7380 = 615*12);
    A = arcsin(sin L * sin 24deg). Sign convention (positive in the present era) is our inference."""
    L = math.radians(360.0*kali_years(jd)/7380.0)
    return -math.degrees(math.asin(math.sin(L)*math.sin(math.radians(24.0))))

TREATISE = {"SuryaNilakantha": treatise_surya_nilakantha, "AryabhataSchool": treatise_aryabhata_school, "DevaAmaraja": treatise_deva_amaraja}

def ayanamsa(jd, model="Lahiri", true=False):
    model = ALIASES.get(model, model)
    if model in MODERN: return _modern(model, jd, true)
    if model in TREATISE: return TREATISE[model](jd)
    raise KeyError(model)

def models(): return list(MODERN) + list(TREATISE)
