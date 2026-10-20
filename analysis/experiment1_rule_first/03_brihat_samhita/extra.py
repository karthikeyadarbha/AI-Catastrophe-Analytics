import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import eclipses as E
W=360/27
q=pd.read_csv("big.csv"); q=q[q.type=="earthquake"].copy(); q["t"]=pd.to_datetime(q.time).dt.tz_localize(None)
sid=pd.read_csv("quakes_m65_sid.csv").set_index("id"); trop=pd.read_csv("quakes_m65_trop.csv").set_index("id")
q=q[q.id.isin(sid.index)].reset_index(drop=True)
# ---- decluster (Gardner-Knopoff style)
q=q.sort_values("mag",ascending=False).reset_index(drop=True)
keep=np.ones(len(q),bool); lat=np.radians(q.latitude.values); lon=np.radians(q.longitude.values); tt=q.t.values.astype("datetime64[s]").astype(np.int64)/86400.0
for i in range(len(q)):
    if not keep[i]: continue
    M=q.mag[i]; T=10**(0.032*M+2.7389); R=10**(0.1238*M+0.983)
    dt=np.abs(tt-tt[i]); 
    d=6371*np.arccos(np.clip(np.sin(lat)*np.sin(lat[i])+np.cos(lat)*np.cos(lat[i])*np.cos(lon-lon[i]),-1,1))
    m=(dt<=T)&(d<=R); m[i]=False; m&=(q.mag.values<=M)
    keep[m]=False
Q=q[keep].sort_values("t").reset_index(drop=True); print("events",len(q),"declustered",len(Q))
MAND={}
for name,l in {"Anila":[14,22,0,11,12,13],"Anala":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]}.items():
    for k in l: MAND[k]=name
SHARP=[5,8,17,18]; SEAS=lambda sun,anchor:(((sun-anchor)%360)//60).astype(int)
def feats(df,frame):
    L={p:df[f"{p}_Lon"].values for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]}
    nk=lambda x:(x//W).astype(int)
    f=pd.DataFrame(index=df.index)
    f["mand"]=[MAND[k] for k in nk(L["Moon"])]
    f["sharp"]=np.isin(nk(L["Moon"]),SHARP)
    f["season"]=SEAS(L["Sun"],22*W if frame=="sid" else 270.0)
    f["jupsat"]=nk(L["Jupiter"])==nk(L["Saturn"])
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: f[p+"_R"]=df[p+"_Is_Retro"].astype(bool).values
    # frame-invariant
    el=(L["Moon"]-L["Sun"])%360; tith=(el//12).astype(int)+1; t15=(tith-1)%15+1
    f["rikta"]=np.isin(t15,[4,9,14]); f["syz"]=np.isin(tith,[29,30,1,14,15,16])
    pl=np.vstack([L[p] for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]]).T
    war=np.zeros(len(df),bool)
    for i in range(5):
        for j in range(i+1,5): war|=np.abs((pl[:,i]-pl[:,j]+180)%360-180)<=3
    f["war3"]=war
    oth=np.vstack([L[p] for p in ["Sun","Mars","Mercury","Jupiter","Venus","Saturn"]]).T
    f["moonpl5"]=(np.abs((L["Moon"][:,None]-oth+180)%360-180)<=5).any(1)
    return f
ev=E.find(1989,2028); ed=np.array([E.dt_to_jd(e["date"]) for e in ev])
def ecl(dtimes):
    jd=np.array([E.dt_to_jd(pd.Timestamp(x).to_pydatetime()) for x in dtimes]); return np.abs(jd[:,None]-ed[None,:]).min(1)
Qs=sid.loc[Q.id].reset_index(); Qt=trop.loc[Q.id].reset_index()
Fs=feats(Qs,"sid"); Ft=feats(Qt,"trop")
bs=pd.read_csv("bg_sid.csv"); bt=pd.read_csv("bg_trop.csv")
Bs=feats(bs,"sid"); Bt=feats(bt,"trop")
Bd=ecl(pd.to_datetime(bs.datetime).dt.tz_localize(None).values); Qd=ecl(Q.t.values)
split=(Q.t>=pd.Timestamp("2008-01-01")).values
# ---- EXTRA (post-hoc, exploratory; family m=4)
def local_hour(df_t,lon): return ((df_t.dt.hour+df_t.dt.minute/60)+lon/15)%24
def jd(t): return 2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds()/86400
def moonlat(t):
    T=(jd(t)-2451545.0)/36525; F=np.radians((93.2720950+483202.0175233*T)%360); return 5.128*np.sin(F)
VAYU=[0,4,6,12,13,14,11]
Qt_=pd.to_datetime(Q.t); Bt_=pd.to_datetime(bs.datetime).dt.tz_localize(None)
lh=local_hour(Qt_,Q.longitude.values)
win=((lh>=3)&(lh<9)).values
nkQ=(Fs_m:=Qs.Moon_Lon.values//W).astype(int); nkB=(bs.Moon_Lon.values//W).astype(int)
vq=np.isin(nkQ,VAYU); pv=np.isin(nkB,VAYU).mean()
rows=[]
def bt(name,o_arr,p0):
    for lab,mask in (("discovery",~split),("holdout",split)):
        n=mask.sum(); o=int(o_arr[mask].sum()); rows.append((name,lab,n,o,round(n*p0,1),round(o/(n*p0),2),stats.binom.sf(o-1,n,p0)))
bt("BS-A Vayu: Moon in 7 Vayu nakshatras",vq,pv)
bt("BS-B Vayu: Moon in set AND local 03-09h",vq&win,pv*0.25)
bt("BS-C local time 03-09h only (control)",win,0.25)
# crescent yoke
elq=(Qs.Moon_Lon-Qs.Sun_Lon).values%360; elb=(bs.Moon_Lon-bs.Sun_Lon).values%360
cq=(elq>=12)&(elq<48); cb=(elb>=12)&(elb<48)
bq=moonlat(Qt_)>0; bb=moonlat(Bt_)>0
print("crescent share ev/bg",cq.mean(),cb.mean(),"north tip share in bg crescents",bb[cb].mean())
# conditional binomial: among crescent-phase events, north-tip share vs bg north-tip share among crescents
p0=bb[cb].mean()
for lab,mask in (("discovery",~split),("holdout",split)):
    m=mask&cq; n=m.sum(); o=int(bq[m].sum()); rows.append(("Ch4 crescent: north tip raised | waxing crescent",lab,n,o,round(n*p0,1),round(o/(n*p0),2),stats.binom.sf(o-1,n,p0)))
bt("Ch4b waxing-crescent phase (12-48 deg)",cq,cb.mean())
R=pd.DataFrame(rows,columns=["test","set","n","observed","expected","ratio","p"]); R["bonf_threshold_m4"]=0.05/4
print(R.round(4).to_string()); R.to_csv("results/bs_vayu_crescent_tests.csv",index=False)
