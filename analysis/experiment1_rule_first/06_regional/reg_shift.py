import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from crescent import *
from reg_helpers import assign
def chunked(tc,lat,lon,n=4000):
    F=[];T=[];A=[]
    for i in range(0,len(tc),n):
        f,t,a=sight(tc[i:i+n],lat[i:i+n],lon[i:i+n]); F.append(f);T.append(t);A.append(a)
    return flags(np.concatenate(F),np.concatenate(T),np.concatenate(A))[0]
Q=pd.read_csv("quakes_m55_input.csv"); A=pd.read_csv("histA_declustered.csv"); B=pd.read_csv("histB_declustered.csv")
jd=np.concatenate([jd_of(pd.to_datetime(Q.datetime).dt.tz_localize(None)),jd_of(pd.to_datetime(A.datetime).dt.tz_localize(None)),jd_of(pd.to_datetime(B.datetime).dt.tz_localize(None))])
lat=np.concatenate([Q.lat.values,A.latitude.values,B.latitude.values]); lon=np.concatenate([Q.lon.values,A.longitude.values,B.longitude.values]); reg=assign(lat,lon)
rng=np.random.default_rng(41); rows=[]
for name in ["Japan-Kurils-Korea","Indonesia"]:
    m=reg==name; j,la,lo=jd[m],lat[m],lon[m]; obs=chunked(conj_before(j),la,lo).mean(); nul=[]
    for _ in range(100): sh=int(rng.integers(-3650,3651)); nul.append(chunked(conj_before(j+sh),la,lo).mean())
    nul=np.array(nul); p=(np.sum(nul>=obs)+1)/101; print(name,"n",m.sum(),"obs",round(obs,4),"null mean",nul.mean().round(4),"sd",nul.std().round(4),"ratio",round(obs/nul.mean(),3),"p",round(p,4),flush=True); rows.append((name,int(m.sum()),obs,nul.mean(),nul.std(),obs/nul.mean(),p))
pd.DataFrame(rows,columns=["region","n_all_periods","observed_rate","shift_null_mean","shift_null_sd","ratio","p_date_shift"]).to_csv("regional_followup_shift.csv",index=False)
