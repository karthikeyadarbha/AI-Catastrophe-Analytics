"""Test: 7 eclipses recorded in inscriptions (Indian Astronomy: A Source Book, 16.38-39). For each stated window we find every syzygy
(full/new moon) with the Moon near a node, by (a) the modern ephemeris `myeph` and (b) this engine (TS parameters, Madhava Rsine),
and compare timing, node distance and the stated weekday. Weekday is the civil day at UT+5:30 (Indian local time)."""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.environ.get("MYEPH_DIR", "."))
import numpy as np
import myeph as M
from indic_engine.compute import planet_positions
from indic_engine.compute.ayanamsa import jd_from_ymd
DAYS = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
def weekday(jd):
    """Indian (sunrise-to-sunrise) weekday: day changes at 06:00 local (UT+5:30), as in the texts."""
    return DAYS[int(math.floor(jd + 0.5 + 5.5/24 - 0.25 + 1)) % 7]
def ist_hhmm(jd):
    f = (jd + 0.5 + 5.5/24) % 1.0; return f"{int(f*24):02d}:{int(f*1440)%60:02d} IST"
def modern_elong(jd): return (M.moon_trop(jd) - M.sun_trop(jd)) % 360
def modern_node_dist(jd): return ((M.moon_trop(jd) - M.node_trop(jd) + 180) % 360) - 180
def indic_state(jd):
    r = planet_positions(jd)["true"]; return (r["Moon"] - r["Sun"]) % 360, ((r["Moon"] - r["MoonNode"] + 180) % 360) - 180, r
def find_syzygies(elong_fn, node_fn, jd0, jd1, target):
    """jd at which elongation == target (0 new moon, 180 full moon)"""
    out = []; step = 0.5; j = jd0; f = lambda x: ((elong_fn(x) - target + 180) % 360) - 180
    prev = f(j)
    while j < jd1:
        nxt = f(j + step)
        if prev < 0 <= nxt and abs(prev) < 40 and abs(nxt) < 40:
            lo, hi = j, j + step
            for _ in range(40):
                mid = 0.5*(lo+hi)
                if f(mid) < 0: lo = mid
                else: hi = mid
            tj = 0.5*(lo+hi); out.append((tj, abs(math.sin(math.radians(node_fn(tj))))))
        prev = nxt; j += step
    return out
def ind_elong(j): return indic_state(j)[0]
def ind_node(j): return indic_state(j)[1]
EVENTS = [  # label, kind(L/S), window start, window end, stated weekday, note
 ("Kalachuri 880 = 1128 CE", "L", (1128,9,25), (1128,12,20), "Thu", "full moon of Karttika, 3rd quarter of night, Moon in Rohini, total"),
 ("Saka 1119 = 1197 CE", "L", (1197,10,20), (1198,1,20), "Sat", "full moon of Margasirsa"),
 ("Saka 1151/52 = 1230-31 CE", "L", (1229,10,20), (1231,1,20), "Thu", "bright half of Margasirsa"),
 ("Kalachuri 322 = 580 CE", "S", (580,3,1), (580,5,15), None, "new moon of Caitra"),
 ("Kalachuri 404 = 662 CE", "S", (662,5,15), (662,8,15), None, "dark half of Asadha"),
 ("Saka 1113 (current year 1191-92; Source Book gives A.D. 1192)", "S", (1191,4,15), (1191,7,20), "Sun", "new moon of Jyestha"),
 ("Saka 1113, book year 1192 window", "S", (1192,4,15), (1192,7,20), "Sun", "same inscription, using the Source Book year as printed"),
 ("Saka 1151/52 = 1230-31 CE", "S", (1229,6,1), (1231,9,1), None, "new moon, Moon/Sun in Karkataka"),
]
def run():
    rows = []
    for lab, kind, a, b, wd, note in EVENTS:
        j0, j1 = jd_from_ymd(*a), jd_from_ymd(*b); tgt = 180.0 if kind == "L" else 0.0
        lim = 0.25 if kind == "L" else 0.32      # |sin(u)| limits: ~ +-14.5 deg / +-18.5 deg node distance (screening, modern)
        mod = [(t, s) for t, s in find_syzygies(modern_elong, modern_node_dist, j0, j1, tgt) if s < lim]
        ind = [(t, s) for t, s in find_syzygies(ind_elong, ind_node, j0, j1, tgt) if s < lim]
        for t, s in mod:
            # match nearest indic syzygy
            best = min(ind, key=lambda x: abs(x[0]-t)) if ind else None
            rows.append(dict(event=lab, kind=kind, modern_jd=t, modern_date=jd_to_date(t), modern_weekday=weekday(t), modern_time=ist_hhmm(t), node_sin=round(s,3), stated_weekday=wd,
                             indic_dt_hours=None if best is None else (best[0]-t)*24, indic_found=best is not None and abs(best[0]-t) < 1.0,
                             indic_weekday=None if best is None else weekday(best[0]), note=note))
        if not mod: rows.append(dict(event=lab, kind=kind, modern_jd=None, modern_date="no eclipse-possible syzygy in window", note=note))
    return rows
def jd_to_date(jd):
    z = int(jd + 0.5); f = jd + 0.5 - z
    a = z if z < 2299161 else z + 1 + int((z - 1867216.25)/36524.25) - int(int((z - 1867216.25)/36524.25)/4)
    b = a + 1524; c = int((b - 122.1)/365.25); d = int(365.25*c); e = int((b - d)/30.6001)
    day = b - d - int(30.6001*e) + f; m = e - 1 if e < 14 else e - 13; y = c - 4716 if m > 2 else c - 4715
    return f"{y:04d}-{m:02d}-{int(day):02d}"
if __name__ == "__main__":
    import csv
    rows = run()
    w = csv.DictWriter(open("inscription_eclipses.csv", "w", newline=""), fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    for r in rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("note",)})
