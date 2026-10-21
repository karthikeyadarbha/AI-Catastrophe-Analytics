"""Scan 1900-2025 for Rohini-sakata-bheda episodes (planet within 1 deg of 17 deg Vrsabha with southern latitude > 2 deg; Brahmagupta rule) in the Indic engine's sky."""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indic_engine.compute import planet_positions, latitude as L, occultation as O
from indic_engine.compute.ayanamsa import jd_from_ymd
hits = []; j = jd_from_ymd(1900,1,1); end = jd_from_ymd(2026,1,1)
while j < end:
    t = planet_positions(j)["true"]
    for p in ("Mars","Jupiter","Saturn","Venus","Mercury"):
        if abs(t[p]-47.0) <= 1.0:
            lat = L.planet_latitude(p, j)
            if O.rohini_sakata_bheda(t[p], lat, 1.0): hits.append((p, j, t[p], lat))
    j += 2.0
eps = []
for h in hits:
    if eps and eps[-1]["planet"] == h[0] and h[1]-eps[-1]["end"] < 40: eps[-1]["end"] = h[1]; eps[-1]["min_lat"] = min(eps[-1]["min_lat"], h[3])
    else: eps.append(dict(planet=h[0], start=h[1], end=h[1], min_lat=h[3]))
def d(j):
    z=int(j+0.5); a=z+1+int((z-1867216.25)/36524.25); a=a-int(int((z-1867216.25)/36524.25)/4); b=a+1524; c=int((b-122.1)/365.25); dd=int(365.25*c); e=int((b-dd)/30.6001)
    day=b-dd-int(30.6001*e); m=e-1 if e<14 else e-13; y=c-4716 if m>2 else c-4715; return f"{y}-{m:02d}-{day:02d}"
with open("rohini_bheda_episodes_1900_2025.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["planet","start","end","most_southern_latitude_deg"])
    for e in eps: w.writerow([e["planet"], d(e["start"]), d(e["end"]), round(e["min_lat"],2)])
print(len(eps), "episodes"); [print(e["planet"], d(e["start"]), d(e["end"]), round(e["min_lat"],2)) for e in eps]
