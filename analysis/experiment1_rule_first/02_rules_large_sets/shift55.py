import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
sys.path.insert(0,"."); import myeph
W=360/27
Q=pd.read_csv("quakes_m55_input.csv"); Q["t"]=pd.to_datetime(Q.datetime).dt.tz_localize(None)
jd0=2440587.5+(Q.t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400; mag=Q.mag.values
def stats_at(jd):
    A=myeph.all_trop(jd); ay=myeph.ayan(jd); L={p:(A[p]-ay)%360 for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]}
    oth=np.vstack([L[p] for p in ["Sun","Mars","Mercury","Jupiter","Venus","Saturn"]]).T
    r10=(np.abs((L["Moon"][:,None]-oth+180)%360-180)<=5).any(1); sharp=np.isin((L["Moon"]//W).astype(int),[5,8,17,18]); vis=(L["Sun"]>=200)&(L["Sun"]<200+W)
    return r10,sharp,vis
new=mag<6.5; b1=(mag>=5.5)&(mag<6.0); b3=mag>=7.0
r10,sh,vi=stats_at(jd0)
obs={"R10 new":r10[new].sum(),"R10 M5.5-5.9":r10[b1].sum(),"sharp new":sh[new].sum(),"Vishakha new":vi[new].sum(),"Vishakha M7+":vi[b3].sum()}
rng=np.random.default_rng(21); res={k:[] for k in obs}
for i in range(1000):
    off=rng.uniform(400,5000)*rng.choice([-1,1]); a,b,c=stats_at(jd0+off)
    res["R10 new"].append(a[new].sum()); res["R10 M5.5-5.9"].append(a[b1].sum()); res["sharp new"].append(b[new].sum()); res["Vishakha new"].append(c[new].sum()); res["Vishakha M7+"].append(c[b3].sum())
for k in obs:
    x=np.array(res[k]); print(k,"observed",obs[k],"shift-null mean",round(x.mean(),1),"sd",round(x.std(),1),"p(>=obs)",round((np.sum(x>=obs[k])+1)/1001,4))
pd.DataFrame({k:res[k] for k in obs}).to_csv("shift_null.csv",index=False)
