"""Compare the Indian-method sky with a modern ephemeris. Reference: any function jd_ut -> {body: sidereal longitude (Lahiri)}.
Default reference in this repo: the astral-ephemeris engine (Swiss Ephemeris) if importable; otherwise pass your own."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from indic_engine.compute.positions import planet_positions
BODIES=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
def wrap(x): return (x+180)%360-180
def compare(jds,ref,param="TS",scheme="madhava"):
    rows=[]
    for jd in jds:
        t=planet_positions(jd,param,scheme)["true"]; r=ref(jd)
        rows.append({"jd":jd,**{b:wrap(t[b]-r[b]) for b in BODIES}})
    return pd.DataFrame(rows)
def summarize(df):
    return pd.DataFrame({"mean_deg":df[BODIES].mean(),"sd_deg":df[BODIES].std(),"max_abs_deg":df[BODIES].abs().max()}).round(3)
