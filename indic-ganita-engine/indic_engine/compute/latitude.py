"""Planetary latitude (viksepa) per Aryabhatiya Ardharatrika / Bhaskara II, Source Book 19.3 (KSS, BG, Brahmagupta KK 1.8.1-6).
nodes (deg): Mars 40, Mercury 20, Jupiter 80, Venus 60, Saturn 100; greatest latitude (arcmin): 90,120,60,120,120.
latitude = greatest_lat * sin(L - node) * R / D, L = manda-sphuta ('true-mean') longitude for Mars/Jupiter/Saturn, sighrocca for Mercury/Venus,
D = (manda hypotenuse * sighra hypotenuse)/R (Earth-planet distance). Moon: 4;30 deg * sin(Moon - node)  [Aryabhatiya; not re-read in this edition]."""
import math
from . import params as P
from .jya import trijya
from . import positions as X
NODE = {"Mars": 40.0, "Mercury": 20.0, "Jupiter": 80.0, "Venus": 60.0, "Saturn": 100.0}
MAXLAT_MIN = {"Mars": 90.0, "Mercury": 120.0, "Jupiter": 60.0, "Venus": 120.0, "Saturn": 120.0}
def moon_latitude(moon_true, moon_node):
    return 4.5*math.sin(math.radians(moon_true - moon_node))
def planet_latitude(planet, jd_ut, param="TS", scheme="madhava"):
    """degrees; positive = north. Distance factor uses the Tantrasangraha hypotenuses (iterated manda karna x sighra karna / R)."""
    m = X.mean_longitudes(jd_ut, param); ms = m["Sun"]; R = trijya(scheme); th = P.MANDOCCA[planet]; mean = m[planet]
    if planet in ("Mars", "Jupiter", "Saturn"):
        P1 = mean - 0.5*X._manda_arc(planet, mean - th, scheme)
        P2 = P1 - X._sighra_exterior(planet, P1, ms, scheme, 0.5)
        P3 = mean - X._manda_arc(planet, P2 - th, scheme)
        Km = X._avisista_karna(planet, P3 - th, scheme)
        c, k = X._SIG[planet]; sg = X._sn(P3 - ms, scheme); rs = R*(c - k*abs(sg))/80
        Ks = math.hypot(R + rs*X._sn(P3 - ms + 90, scheme), rs*sg)
        L = P3
    else:
        kap = mean - th; Km = X._avisista_karna(planet, kap, scheme); th_ms = mean - X._manda_arc(planet, kap, scheme)
        sg = X._sn(th_ms - ms, scheme); c, k = X._SIG[planet]; sphg = (c - k*abs(sg))*Km/R
        Ks = math.hypot(R + R*X._sn(th_ms - ms + 90, scheme)*sphg/80, R*sg*sphg/80); L = th_ms    # sighrocca = the planet's own heliocentric (manda-corrected) longitude
    D = Km*Ks/R
    lat_min = MAXLAT_MIN[planet]*math.sin(math.radians(L - NODE[planet]))*R/D
    return lat_min/60.0
