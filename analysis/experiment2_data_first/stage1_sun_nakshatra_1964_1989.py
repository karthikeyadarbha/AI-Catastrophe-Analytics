import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import numpy as np,pandas as pd
from scipy import stats
def sun_trop(t):
    jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds()/86400; T=(jd-2451545.0)/36525
    L0=280.46646+36000.76983*T; M=np.radians(357.52911+35999.05029*T)
    C=(1.914602-0.004817*T)*np.sin(M)+0.019993*np.sin(2*M)+0.000289*np.sin(3*M)
    return (L0+C-0.00569)%360
def ayan(t): return 23.853+0.01397*((t.dt.year+t.dt.dayofyear/365.25)-2000)
def sid(t): return (sun_trop(t)-ayan(t))%360
# validate vs tool on 1990+ events
q=pd.read_csv("quakes_m65_sid.csv"); big=pd.read_csv("big.csv").set_index("id")
tt=pd.to_datetime(big.loc[q.id].time).dt.tz_localize(None).reset_index(drop=True)
print("max |diff| vs tool (deg):",np.abs(((sid(tt)-q.Sun_Lon.values+180)%360)-180).max().round(3))
d=pd.read_csv("old_m65.csv"); d=d[d.type=="earthquake"].copy(); d["t"]=pd.to_datetime(d.time).dt.tz_localize(None)
d=d.sort_values("mag",ascending=False).reset_index(drop=True)
keep=np.ones(len(d),bool); lat=np.radians(d.latitude.values); lon=np.radians(d.longitude.values); tm=d.t.values.astype("datetime64[s]").astype(np.int64)/86400.0
for i in range(len(d)):
    if not keep[i]: continue
    M=d.mag[i]; T=10**(0.032*M+2.7389); R=10**(0.1238*M+0.983)
    dist=6371*np.arccos(np.clip(np.sin(lat)*np.sin(lat[i])+np.cos(lat)*np.cos(lat[i])*np.cos(lon-lon[i]),-1,1))
    m=(np.abs(tm-tm[i])<=T)&(dist<=R); m[i]=False; m&=(d.mag.values<=M); keep[m]=False
D=d[keep].reset_index(drop=True); print("old events",len(d),"declustered",len(D))
days=pd.Series(pd.date_range("1964-01-01","1989-12-31",freq="6h")); p0=((sid(days)>=200)&(sid(days)<213.3333)).mean()
s=sid(D.t); inw=((s>=200)&(s<213.3333)).values
n=len(D); o=int(inw.sum()); print("Vishakha window share",round(p0,4),"n",n,"obs",o,"exp",round(n*p0,1),"ratio",round(o/(n*p0),2),"p",round(stats.binom.sf(o-1,n,p0),4))
m7=D.mag.values>=7.0; o7=int(inw[m7].sum()); print("M>=7",m7.sum(),o7,round(m7.sum()*p0,1),round(stats.binom.sf(o7-1,m7.sum(),p0),4))
# also pooled Sun-nakshatra profile for old set
nk=(s//(360/27)).astype(int); bn=(sid(days)//(360/27)).astype(int); pb=np.bincount(bn,minlength=27)/len(bn); ob=np.bincount(nk,minlength=27)
print(np.round(ob/(pb*n),2).tolist())
pd.DataFrame({"nak":range(27),"obs":ob,"exp":(pb*n).round(1)}).to_csv("results/sun_nakshatra_1964_1989_followup.csv",index=False)
