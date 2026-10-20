import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys, random, datetime as dt
sys.path.insert(0,".")
import eclipses as E, pandas as pd, numpy as np
df=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv")
df["t"]=pd.to_datetime(df.UTC)
ev=E.find(1996,2028)
et=[(pd.Timestamp(e["date"]),e["kind"]) for e in ev]
def mind(t,kind=None):
    d=[abs((x-t).total_seconds())/86400 for x,k in et if kind in (None,k)]
    return min(d)
def signed(t):
    return min(((x-t).total_seconds()/86400 for x,k in et),key=abs)
df["d_any"]=[mind(t) for t in df.t]
df["d_solar"]=[mind(t,"solar") for t in df.t]
df["d_lunar"]=[mind(t,"lunar") for t in df.t]
df["signed"]=[signed(t) for t in df.t]
# 5.7: eclipse ~24 months after quake (+/-1 month)
def has24(t):
    return any(abs((x-t).days-730)<=30 for x,k in et)
df["ecl_24m"]=[has24(t) for t in df.t]
rng=np.random.default_rng(1)
ed=np.array([x.value/8.64e13 for x,k in et])
d0=pd.Timestamp("1999-09-01").value/8.64e13
span=int(pd.Timestamp("2024-06-01").value/8.64e13-d0)
N=20000
days=d0+rng.integers(0,span,(N,15))
m=np.abs(days[:,:,None]-ed[None,None,:]).min(2)
res={w:(m<=w).sum(1) for w in (7,15,30)}
r24=(np.abs((ed[None,None,:]-days[:,:,None])-730)<=30).any(2).sum(1)
out=[]
for w in (7,15,30):
    obs=int((df.d_any<=w).sum()); b=np.array(res[w])
    out.append((f"within ±{w}d of any eclipse",obs,b.mean(),(b>=obs).mean()))
obs=int(df.ecl_24m.sum()); b=np.array(r24)
out.append(("eclipse 24±1 mo after quake (5.7)",obs,b.mean(),(b>=obs).mean()))
print(pd.DataFrame(out,columns=["test","observed_of_15","MC_mean","p(>=obs)"]).round(3).to_string())
print(df[["Event","UTC","d_any","d_solar","d_lunar","signed","ecl_24m"]].round(1).to_string())
df[["Event","UTC","d_any","d_solar","d_lunar","signed","ecl_24m"]].round(2).to_csv("results/eclipse_proximity_results.csv",index=False)
print(np.corrcoef(df.d_any,np.log10(df.Deaths))[0,1])
