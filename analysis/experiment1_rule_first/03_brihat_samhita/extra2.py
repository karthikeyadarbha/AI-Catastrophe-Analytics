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
BS={}
for name,l in {"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]}.items():
    for k in l: BS[k]=name
assert len(BS)==27
levels=["Vayu","Agni","Indra","Varuna"]
Qt_=pd.to_datetime(Q.t)
lh=(((Qt_.dt.hour+Qt_.dt.minute/60)+Q.longitude.values/15)%24).values
tq=np.where((lh>=18),"Indra",np.where(lh<6,"Varuna",np.where(lh<12,"Vayu","Agni")))
rows=[]
for fr,Qf,Bf in (("sid",Qs,bs),("trop",Qt,bt)):
    nkQ=(Qf.Moon_Lon.values//W).astype(int); nkB=(Bf.Moon_Lon.values//W).astype(int)
    mq=np.array([BS[k] for k in nkQ]); mb=np.array([BS[k] for k in nkB])
    sh=np.array([(mb==l).mean() for l in levels])
    for lab,mask in (("discovery",~split),("holdout",split)):
        obs=np.array([(mq[mask]==l).sum() for l in levels]); n=obs.sum(); c,p=stats.chisquare(obs,sh*n)
        rows.append((f"BS-M mandala by Moon nakshatra [{fr}]",lab,n,"/".join(map(str,obs)),"/".join(f"{x:.0f}" for x in sh*n),p))
    if fr=="sid":
        # concordance with time-of-day mandala (Gargya halves of day/night, local solar time)
        ps=np.array([(tq==l).mean() for l in levels]); p0=float((sh*ps).sum())
        for lab,mask in (("discovery",~split),("holdout",split)):
            n=mask.sum(); o=int((mq[mask]==tq[mask]).sum()); rows.append(("BS-T nakshatra mandala = time-of-day mandala",lab,n,o,round(n*p0,1),stats.binom.sf(o-1,n,p0)))
R=pd.DataFrame(rows,columns=["test","set","n","observed","expected","p"]); R["bonf_threshold_m3"]=0.05/3
pd.set_option("display.width",220); print(R.to_string()); R.to_csv("results/bs_ch32_mandala_tests.csv",index=False)
