import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import pandas as pd, numpy as np
from scipy.optimize import minimize_scalar
d=pd.read_csv("usgs_catalogue_15_events.csv"); d["t"]=pd.to_datetime(d.time).dt.tz_localize(None)
ev=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv"); ev["t0"]=pd.to_datetime(ev.UTC)
LAGS=np.array([3,4,7,14,30,45.0]); TOL=0.5; TMAX=90.0; C=0.05
rng=np.random.default_rng(11)
def omori_sample(n,p,tmax=TMAX,c=C):
    u=rng.random(n)
    if abs(p-1)<1e-6: return c*((tmax+c)/c)**u-c
    a=1-p; return ((u*((tmax+c)**a-c**a)+c**a))**(1/a)-c
def fit_p(t):
    def nll(p):
        a=1-p
        Z=np.log(((TMAX+C)**a-C**a)/a) if abs(a)>1e-9 else np.log(np.log((TMAX+C)/C))
        return len(t)*Z+p*np.sum(np.log(t+C))
    return minimize_scalar(nll,bounds=(0.3,2.0),method="bounded").x
def hits(t):  # number of verse lags with >=1 event within +/-TOL days
    return int(sum(np.any(np.abs(t-L)<=TOL) for L in LAGS))
def gaps_hits(t):  # successive inter-event gaps matching lags
    g=np.diff(np.sort(t)); return int(sum(np.any(np.abs(g-L)<=TOL) for L in LAGS))
summary=[]; N=4000
for M in (5.0,4.5,5.5):
    rows=[]
    for r in ev.itertuples():
        x=d[d.MainShock==r.Event]
        # mainshock = largest event within 1 day of listed time
        near=x[(x.t-r.t0).abs()<pd.Timedelta(days=1)]
        ms=near.loc[near.mag.idxmax()]; t0=ms.t
        a=x[(x.t>t0+pd.Timedelta(seconds=1))&(x.t<=t0+pd.Timedelta(days=TMAX))&(x.mag>=M)&(x.id!=ms.id)]
        ta=((a.t-t0).dt.total_seconds()/86400).values
        fore=x[(x.t<t0-pd.Timedelta(seconds=1))&(x.mag>=M)]
        n=len(ta)
        if n<2: rows.append((r.Event,n,len(fore),np.nan,np.nan,np.nan,np.nan,np.nan));continue
        p=fit_p(ta); h=hits(ta); gh=gaps_hits(ta)
        sims=np.array([hits(omori_sample(n,p)) for _ in range(N)]); gs=np.array([gaps_hits(omori_sample(n,p)) for _ in range(N)])
        rows.append((r.Event,n,len(fore),round(p,2),h,round(sims.mean(),2),gh,round(gs.mean(),2)))
    R=pd.DataFrame(rows,columns=["Event","n_aftershocks","n_foreshocks_60d","omori_p","lag_hits_obs","lag_hits_omori_mean","gap_hits_obs","gap_hits_omori_mean"])
    ok=R.dropna()
    # pooled permutation: simulate whole set
    print(f"\n=== M>={M} ===");print(R.to_string())
    tot_o=ok.lag_hits_obs.sum(); tot_e=ok.lag_hits_omori_mean.sum()
    # pooled p via simulation
    simtot=np.zeros(N)
    for r in ev.itertuples():
        pass
    summary.append((M,len(ok),int(tot_o),round(tot_e,1),int(ok.gap_hits_obs.sum()),round(ok.gap_hits_omori_mean.sum(),1)))
    R.to_csv(f"verse18_7_results_M{M}.csv",index=False)
print(pd.DataFrame(summary,columns=["Mmin","events_used","lag_hits_obs_total","lag_hits_omori_expected","gap_hits_obs","gap_hits_omori_expected"]).to_string())
