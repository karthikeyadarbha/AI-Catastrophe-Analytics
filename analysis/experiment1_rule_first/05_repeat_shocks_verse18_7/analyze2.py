import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import pandas as pd, numpy as np
from scipy.optimize import minimize_scalar
d=pd.read_csv("usgs_catalogue_15_events.csv"); d["t"]=pd.to_datetime(d.time).dt.tz_localize(None)
ev=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv"); ev["t0"]=pd.to_datetime(ev.UTC)
LAGS=np.array([3,4,7,14,30,45.0]); TOL=0.5; TMIN=0.5; TMAX=90.0; C=0.05
rng=np.random.default_rng(21)
def samp(n,p):
    u=rng.random(n); a=1-p
    if abs(a)<1e-6: return C*np.exp(u*np.log((TMAX+C)/(TMIN+C)))*((TMIN+C)/C)-C
    lo,hi=(TMIN+C)**a,(TMAX+C)**a
    return (u*(hi-lo)+lo)**(1/a)-C
def fit(t):
    def nll(p):
        a=1-p; Z=np.log(((TMAX+C)**a-(TMIN+C)**a)/a) if abs(a)>1e-9 else np.log(np.log((TMAX+C)/(TMIN+C)))
        return len(t)*Z+p*np.sum(np.log(t+C))
    return minimize_scalar(nll,bounds=(0.3,2.0),method="bounded").x
hits=lambda t:int(sum(np.any(np.abs(t-L)<=TOL) for L in LAGS))
gh=lambda t:int(sum(np.any(np.abs(np.diff(np.sort(t))-L)<=TOL) for L in LAGS))
N=3000; out=[]
for M in (5.0,4.5):
    obs_l=obs_g=0; simL=np.zeros(N); simG=np.zeros(N); used=0
    for r in ev.itertuples():
        x=d[d.MainShock==r.Event]; near=x[(x.t-r.t0).abs()<pd.Timedelta(days=1)]; ms=near.loc[near.mag.idxmax()]; t0=ms.t
        a=x[(x.t>t0)&(x.mag>=M)&(x.id!=ms.id)]; ta=((a.t-t0).dt.total_seconds()/86400).values; ta=ta[(ta>=TMIN)&(ta<=TMAX)]
        if len(ta)<5: continue
        used+=1; p=fit(ta); obs_l+=hits(ta); obs_g+=gh(ta)
        for k in range(N):
            s=samp(len(ta),p); simL[k]+=hits(s); simG[k]+=gh(s)
    out.append((M,used,obs_l,round(simL.mean(),1),round((simL>=obs_l).mean(),3),round((simL<=obs_l).mean(),3),obs_g,round(simG.mean(),1),round((simG>=obs_g).mean(),3)))
R=pd.DataFrame(out,columns=["Mmin","events_used(>=5 aftershocks after 0.5d)","lag_hits_obs","lag_hits_expected_Omori","p_obs>=exp","p_obs<=exp","gap_hits_obs","gap_hits_expected","p_gap_obs>=exp"])
print(R.to_string()); R.to_csv("results/verse18_7_completeness_corrected.csv",index=False)
