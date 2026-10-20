import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import myeph as M
from hist import grid
def lonlat(name,jd):
    T=(jd-2451545.0)/36525; X,Y,Z=M.helio(name,T); Xe,Ye,Ze=M.helio("Earth",T)
    dx,dy,dz=X-Xe,Y-Ye,Z-Ze; return np.degrees(np.arctan2(dy,dx))%360, np.degrees(np.arctan2(dz,np.hypot(dx,dy)))
def cond(t):
    jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
    lm,bm=lonlat("Mercury",jd); lj,bj=lonlat("Jupiter",jd)
    dl=((lm-lj+180)%360)-180; sep=np.hypot(dl*np.cos(np.radians((bm+bj)/2)),bm-bj); south=bm<bj
    return {"C1 Mercury-Jupiter within 1 deg (planetary war)":sep<1,"C2 same, Mercury defeated (the southern one)":(sep<1)&south,
            "C3 within 5 deg":sep<5,"C4 within 5 deg, Mercury the southern one":(sep<5)&south}
rows=[]
def run(label,t,tb):
    rng=np.random.default_rng(11); tb=(tb+pd.to_timedelta(rng.uniform(0,6*3600,len(tb)),unit='s')).reset_index(drop=True)
    ce,cb=cond(t),cond(tb)
    for k in ce:
        o=int(ce[k].sum()); p0=cb[k].mean(); n=len(t); ex=n*p0
        rows.append((label,k,n,o,round(ex,1),round(o/ex,3) if ex else np.nan,stats.binom.sf(o-1,n,p0)))
Q=pd.read_csv("quakes_m55_input.csv"); bg=pd.read_csv("bg_sid.csv")
tq=pd.to_datetime(Q.datetime).dt.tz_localize(None); tb=pd.to_datetime(bg.datetime).dt.tz_localize(None).reset_index(drop=True)
sp=(tq>=pd.Timestamp("2008-01-01")).values; bsp=(tb>=pd.Timestamp("2008-01-01")).values
run("Modern M5.5+ all",tq,tb); run("Modern M5.5+ 1990-2007 discovery",tq[~sp].reset_index(drop=True),tb[~bsp].reset_index(drop=True)); run("Modern M5.5+ 2008-2026 holdout",tq[sp].reset_index(drop=True),tb[bsp].reset_index(drop=True))
for lab,f,y0,y1 in (("Older set 1 (1964-1989)","histA_declustered.csv",1964,1989),("Older set 2 (1900-1963)","histB_declustered.csv",1900,1963)):
    Qh=pd.read_csv(f); th=pd.to_datetime(Qh.datetime).dt.tz_localize(None); G=grid(y0,y1); run(lab,th,G.datetime.reset_index(drop=True))
R=pd.DataFrame(rows,columns=["data_set","test","n","observed","expected","ratio","p"]); R["threshold"]=0.0125; R["passes"]=R.p<0.0125
pd.set_option("display.width",230); print(R.round(4).to_string()); R.to_csv("results/brihat_samhita_ch17_mercury_jupiter_results.csv",index=False)
