import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import pandas as pd,numpy as np,itertools
rng=np.random.default_rng(7)
d=pd.read_csv('aftershocks_M7plus.csv')
d['t']=(pd.to_datetime(d.time,utc=True)-pd.to_datetime(d.MS_time,utc=True)).dt.total_seconds()/86400
d=d[(d.t>0.02)]
C=0.05
def fit_p(t,lo,hi):
    best=None
    for p in np.arange(0.5,1.8,0.02):
        a=1-p
        if abs(a)<1e-6: a=1e-6
        # density ∝ (t+C)^-p on [lo,hi]
        Z=((hi+C)**a-(lo+C)**a)/a
        ll=-p*np.log(t+C).sum()-len(t)*np.log(Z)
        if best is None or ll>best[0]: best=(ll,p)
    return best[1]
def samp(n,p,lo,hi):
    a=1-p
    if abs(a)<1e-6: a=1e-6
    u=rng.random(n)
    return (u*((hi+C)**a-(lo+C)**a)+(lo+C)**a)**(1/a)-C
def run(minmag,LAGS,TOL,TMIN=0.5,TMAX=90.,nsim=300):
    LAGS=np.array(LAGS,float)
    obs_l=obs_g=0; sims_l=np.zeros(nsim); sims_g=np.zeros(nsim); n_ev=0
    for ms,g in d[d.mag>=minmag].groupby('MainShock'):
        t=np.sort(g.t.values); t=t[(t>=TMIN)&(t<=TMAX)]
        if len(t)<4: continue
        n_ev+=len(t); p=fit_p(t,TMIN,TMAX)
        hl=lambda x:sum(np.any(np.abs(x-L)<=TOL) for L in LAGS)
        hg=lambda x:sum(np.any(np.abs(np.diff(np.sort(x))-L)<=TOL) for L in LAGS)
        obs_l+=hl(t); obs_g+=hg(t)
        for i in range(nsim):
            s=samp(len(t),p,TMIN,TMAX); sims_l[i]+=hl(s); sims_g[i]+=hg(s)
    pl=(np.sum(sims_l>=obs_l)+1)/(nsim+1); pg=(np.sum(sims_g>=obs_g)+1)/(nsim+1)
    return n_ev,obs_l,sims_l.mean(),pl,obs_g,sims_g.mean(),pg
rows=[]
for mm in (5.0,5.5):
  for tol in (0.25,0.5,1.0):
    for name,lags in (('3,4,7,14,30,45',[3,4,7,14,30,45]),('3,4,7,15,30,45',[3,4,7,15,30,45]),('3,4,7 only',[3,4,7]),('14,30,45 only',[14,30,45])):
        r=run(mm,lags,tol,nsim=150)
        rows.append((mm,tol,name)+r); print(rows[-1],flush=True)
pd.DataFrame(rows,columns='minmag tol lags n_events lag_obs lag_exp lag_p gap_obs gap_exp gap_p'.split()).to_csv('verse18_7_M7plus_robustness.csv',index=False)
