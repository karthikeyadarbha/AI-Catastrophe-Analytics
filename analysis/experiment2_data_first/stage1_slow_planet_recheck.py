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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P7=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
def cats(df):
    L={p:df[f"{p}_Lon"].values for p in P7}; C={}
    for p in P7: C[f"{p} nakshatra"]=(L[p]//W).astype(int),27
    for p in P7: C[f"{p} sign"]=(L[p]//30).astype(int),12
    C["Moon tithi (30)"]=(((L["Moon"]-L["Sun"])%360)//12).astype(int),30
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]:
        C[f"Moon-{p} separation"]=(((L["Moon"]-L[p])%360)//30).astype(int),12
        C[f"Sun-{p} separation"]=(((L["Sun"]-L[p])%360)//30).astype(int),12
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: C[f"{p} retro"]=df[p+"_Is_Retro"].astype(int).values,2
    return C
CQ=cats(Qs); CB=cats(bs)
bt_=pd.to_datetime(bs.datetime).dt.tz_localize(None); bsplit=(bt_>=pd.Timestamp("2008-01-01")).values
def zs(CQ,mask,CB,bmask=None):
    out={}
    for k,(a,m) in CQ.items():
        bb=CB[k][0] if bmask is None else CB[k][0][bmask]
        pb=np.bincount(bb,minlength=m)/len(bb); o=np.bincount(a[mask],minlength=m); e=pb*mask.sum()
        out[k]=(o-e)/np.sqrt(np.maximum(e*(1-pb),1e-9))
    return out
zd=zs(CQ,~split,CB,~bsplit); zh=zs(CQ,split,CB,bsplit)
keys=list(CQ); Zd=np.concatenate([zd[k] for k in keys]); Zh=np.concatenate([zh[k] for k in keys])
r_obs=np.corrcoef(Zd,Zh)[0,1]; print("bins",len(Zd),"corr z_disc,z_hold",round(r_obs,3))
# null: draw events from background, same sizes
rng=np.random.default_rng(11); nd=int((~split).sum()); nh=int(split.sum()); nulls=[]
bi_d=np.where(~bsplit)[0]; bi_h=np.where(bsplit)[0]
for _ in range(300):
    idx=np.concatenate([rng.choice(bi_d,nd),rng.choice(bi_h,nh)]); CQn={k:(v[0][idx],v[1]) for k,v in CB.items()}
    m=np.zeros(nd+nh,bool); m[nd:]=True
    a_=zs(CQn,~m,CB,~bsplit); b_=zs(CQn,m,CB,bsplit); nulls.append(np.corrcoef(np.concatenate([a_[k] for k in keys]),np.concatenate([b_[k] for k in keys]))[0,1])
nulls=np.array(nulls); print("null corr mean/sd",nulls.mean().round(3),nulls.std().round(3),"p(>=obs)",(nulls>=r_obs).mean())
# replication of top discovery bins
rows=[(k,i,zd[k][i],zh[k][i]) for k in keys for i in range(len(zd[k]))]
D=pd.DataFrame(rows,columns=["feature","bin","z_disc","z_hold"])
top=D.reindex(D.z_disc.abs().sort_values(ascending=False).index).head(20); print(top.round(2).to_string())
sel=D[D.z_disc>1.96]; print("disc bins z>1.96:",len(sel),"hold same-dir z>1.96:",(sel.z_hold>1.96).sum(),"mean hold z",sel.z_hold.mean().round(2))
# era consistency for Moon nakshatra
eras=pd.cut(Q.t.dt.year,[1989,2002,2014,2027],labels=False).values
fig,ax=plt.subplots(2,2,figsize=(10,7),dpi=150)
nb=len(bs); a=ax[0,0]; k="Moon nakshatra"; pb=np.bincount(CB[k][0],minlength=27)/nb
for e,(c,l) in enumerate(zip(["#9ecae1","#4292c6","#08519c"],["1990-2002","2003-2014","2015-2026"])):
    m=eras==e; o=np.bincount(CQ[k][0][m],minlength=27); a.plot(range(27),o/(pb*m.sum()),"-o",ms=3,color=c,lw=1,label=l)
a.axhline(1,color="k",lw=.6); a.set_title("Moon nakshatra: events / expected, three eras",fontsize=9); a.set_xlabel("nakshatra index (0=Ashwini)"); a.legend(fontsize=7)
a=ax[0,1]; k="Moon tithi (30)"; pb=np.bincount(CB[k][0],minlength=30)/nb
o=np.bincount(CQ[k][0],minlength=30); a.bar(range(30),o/(pb*len(Q)),color="#4292c6"); a.axhline(1,color="k",lw=.6); a.set_title("Moon tithi (1=new moon ... 15=full ... 30): events / expected",fontsize=9); a.set_xlabel("tithi - 1")
a=ax[1,0]; a.scatter(Zd,Zh,s=6,alpha=.5,color="#2a5d9f"); a.axhline(0,color="k",lw=.5); a.axvline(0,color="k",lw=.5); a.set_xlabel("z, discovery (before 2008)"); a.set_ylabel("z, holdout (2008+)"); a.set_title(f"All {len(Zd)} bins: discovery vs holdout (r={r_obs:.2f}; null {nulls.mean():.2f}±{nulls.std():.2f})",fontsize=9)
a=ax[1,1]; a.hist(nulls,bins=25,color="#bbb"); a.axvline(r_obs,color="#c0392b"); a.set_title("Chance distribution of r (300 resamples); red = observed",fontsize=9)
fig.tight_layout(); fig.savefig("/tmp/claude-0/-home-claude/d170e3d9-100e-58bf-b885-3cf047c3f89f/scratchpad/work/bottomup.png")
D.to_csv("results/bottom_up_bin_scan.csv",index=False)
# by magnitude, depth, region subsets: Moon-nakshatra mandala and sharp group


# recheck slow-planet prereg rules with period-matched baselines
rows=[]
def bt(name,ev,bg):
    for lab,m,bm in (("discovery",~split,~bsplit),("holdout",split,bsplit)):
        p0=bg[bm].mean(); n=int(m.sum()); o=int(ev[m].sum()); rows.append((name,lab,n,o,round(n*p0,1),round(o/(n*p0),2),stats.binom.sf(o-1,n,p0)))
for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: bt(f"R7 {p} retrograde [matched]",CQ[f"{p} retro"][0]==1,CB[f"{p} retro"][0]==1)
bt("R8 Jupiter=Saturn nakshatra [sid, matched]",CQ["Jupiter nakshatra"][0]==CQ["Saturn nakshatra"][0],CB["Jupiter nakshatra"][0]==CB["Saturn nakshatra"][0])
def pair(df):
    L=np.vstack([df[f"{p}_Lon"].values for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]]).T; w=np.zeros(len(df),bool)
    for a in range(5):
        for b in range(a+1,5): w|=np.abs((L[:,a]-L[:,b]+180)%360-180)<=3
    return w
bt("R9 planet pair within 3 deg [matched]",pair(Qs),pair(bs))
R=pd.DataFrame(rows,columns=["test","set","n","observed","expected","ratio","p"]); R["bonf_threshold_23"]=0.05/23
print(R.round(4).to_string()); R.to_csv("results/prereg_slow_planet_recheck_matched_baseline.csv",index=False)
