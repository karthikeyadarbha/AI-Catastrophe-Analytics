"""Dṛggaṇita (Parameśvara, 1431 CE) against a modern ephemeris, century by century, with the Tantrasaṅgraha (TS) for comparison.
Reference: myeph (low-precision VSOP-type theory used throughout this project) minus the Lahiri ayanāṃśa.
Error = Indic - modern, degrees. Every Indic frame uses its own sidereal year, so the Sun's error drifts about -0.24 deg per century;
compare planets with the Sun's error in the same row."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indic_engine.compute import drk, positions as P, ayanamsa as A
BODIES=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
def wrap(x): return (x+180)%360-180
def run(myeph, y0=1000, y1=2200, step=23.0):
    rows=[]
    for j in np.arange(2451545.0+(y0-2000)*365.25, 2451545.0+(y1-2000)*365.25, step):
        a=A.ayanamsa(j,"Lahiri"); tr=myeph.all_trop(j); ref={b:(float(tr[b])-a)%360 for b in BODIES}
        d=drk.planet_positions(j)["true"]; t=P.planet_positions(j,"TS")["true"]; r={"year":2000+(j-2451545.0)/365.25}
        for b in BODIES: r["Drk_"+b]=wrap(d[b]-ref[b]); r["TS_"+b]=wrap(t[b]-ref[b])
        rows.append(r)
    o=pd.DataFrame(rows); o["century"]=(o.year//100)*100; return o
if __name__=="__main__":
    sys.path.insert(0,sys.argv[1]); import myeph
    o=run(myeph); o.to_csv(os.path.join(os.path.dirname(__file__),"drk_vs_modern_1000_2200.csv"),index=False)
    for pre in ("Drk_","TS_"):
        print(pre); print(o.groupby("century")[[pre+b for b in BODIES]].agg(["mean","std"]).round(2).to_string())
