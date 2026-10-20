import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import myeph as M
W=360/27; P7=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
def decluster(q):
    q=q.sort_values("mag",ascending=False).reset_index(drop=True)
    q["t"]=pd.to_datetime(q.time).dt.tz_localize(None)
    keep=np.ones(len(q),bool); lat=np.radians(q.latitude.values); lon=np.radians(q.longitude.values); tt=q.t.values.astype("datetime64[s]").astype(np.int64)/86400.0; mg=q.mag.values
    sl,cl=np.sin(lat),np.cos(lat)
    for i in range(len(q)):
        if not keep[i]: continue
        m_=mg[i]; T=10**(0.032*m_+2.7389) if m_>=6.5 else 10**(0.5409*m_-0.547); R=10**(0.1238*m_+0.983)
        m=(np.abs(tt-tt[i])<=T)&(mg<=m_); m[i]=False; m&=keep; idx=np.where(m)[0]
        if len(idx)==0: continue
        d=6371*np.arccos(np.clip(sl[idx]*sl[i]+cl[idx]*cl[i]*np.cos(lon[idx]-lon[i]),-1,1)); keep[idx[d<=R]]=False
    Q=q[keep].sort_values("t").reset_index(drop=True); Q["datetime"]=Q.t.dt.strftime("%Y-%m-%dT%H:%M:%SZ"); return Q
def grid(y0,y1): return pd.DataFrame({"datetime":pd.date_range(f"{y0}-01-01",f"{y1}-12-31 18:00",freq="6h")})
def trop(d,ay):
    d=d.copy()
    for p in P7: d[p+"_Lon"]=(d[p+"_Lon"]+ay)%360
    return d
