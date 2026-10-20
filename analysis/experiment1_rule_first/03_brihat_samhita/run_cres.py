import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from scipy import stats
from crescent import *
def chunked(tc,lat,lon,n=4000):
    F=[];T=[];A=[]
    for i in range(0,len(tc),n):
        f,t,a=sight(tc[i:i+n],lat[i:i+n],lon[i:i+n]); F.append(f);T.append(t);A.append(a)
    return np.concatenate(F),np.concatenate(T),np.concatenate(A)
# Ujjain monthly table 1899-2028
k=np.arange(int((1899-2000)*12.3685),int((2029-2000)*12.3685)); tcm=conj_before(2451550.09766+29.530588861*k+1.0); tcm=np.sort(tcm)
f,t,a=chunked(tcm,np.full(len(tcm),23.18),np.full(len(tcm),75.78)); Yu,Lu=flags(f,t,a); print("Ujjain months",len(tcm),"yoke share",Yu.mean().round(3),"level share",Lu.mean().round(3))
def lookup(jd): return np.searchsorted(tcm,jd,side="right")-1
rng=np.random.default_rng(17)
sets=[]
Q=pd.read_csv("quakes_m55_input.csv"); tq=pd.to_datetime(Q.datetime).dt.tz_localize(None)
sets.append(("1990-2026 all",jd_of(tq),Q.lat.values,Q.lon.values,(1990,2026)))
sp=(tq>=pd.Timestamp("2008-01-01")).values
sets.append(("2008-2026 holdout",jd_of(tq)[sp],Q.lat.values[sp],Q.lon.values[sp],(2008,2026)))
sets.append(("1990-2007 discovery",jd_of(tq)[~sp],Q.lat.values[~sp],Q.lon.values[~sp],(1990,2007)))
for lab,fn,y0,y1 in (("Era A 1964-1989","histA_declustered.csv",1964,1989),("Era B 1900-1963 (M7+)","histB_declustered.csv",1900,1963)):
    h=pd.read_csv(fn); sets.append((lab,jd_of(pd.to_datetime(h.datetime).dt.tz_localize(None)),h.latitude.values,h.longitude.values,(y0,y1)))
rows=[]
for lab,jd,lat,lon,(y0,y1) in sets:
    for nm,flag in (("yoke",Yu),("level",Lu)):
        idx=lookup(jd); obs=flag[idx].mean(); nul=[]
        for _ in range(500):
            sh=int(rng.integers(-3650,3651)); j2=np.clip(jd+sh,tcm[0],tcm[-1]); nul.append(flag[lookup(j2)].mean())
        nul=np.array(nul); p=(np.sum(nul>=obs)+1)/501; rows.append((lab,f"U {nm} (Ujjain, date-shift null)",len(jd),round(obs,4),round(nul.mean(),4),round(obs/nul.mean(),3),p))
    # epicentre
    tc=conj_before(jd); fe,te,ae=chunked(tc,lat,lon); ye,le=flags(fe,te,ae)
    G=grid(y0,y1); gj=jd_of(G.datetime); bgY=[];bgL=[]
    for d in range(3):
        ii=rng.integers(0,len(gj),len(jd)); pm=rng.permutation(len(jd)); tcb=conj_before(gj[ii]); fb,tb_,ab=chunked(tcb,lat[pm],lon[pm]); yb,lb=flags(fb,tb_,ab); bgY.append(yb);bgL.append(lb)
    for nm,ev,bg in (("yoke",ye,np.concatenate(bgY)),("level",le,np.concatenate(bgL))):
        n=len(ev); o=int(ev.sum()); p0=bg.mean(); e=n*p0; rows.append((lab,f"L {nm} (epicentre, binomial)",n,o,round(e,1),round(o/e,3),stats.binom.sf(o-1,n,p0)))
    print(lab,"done",flush=True)
Rr=pd.DataFrame(rows,columns=["data_set","test","n","observed","expected","ratio","p"]); Rr["threshold"]=0.0125; Rr["passes"]=Rr.p<0.0125
pd.set_option("display.width",220); print(Rr.round(4).to_string()); Rr.to_csv("crescent_results.csv",index=False)
