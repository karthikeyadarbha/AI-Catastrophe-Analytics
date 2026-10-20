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
rows=[]
def binom_test(name,col_ev,col_bg):
    p0=col_bg.mean()
    for lab,mask in (("discovery",~split),("holdout",split)):
        n=mask.sum(); o=int(col_ev[mask].sum()); p=stats.binom.sf(o-1,n,p0)
        rows.append((name,lab,n,o,round(n*p0,1),round(o/(n*p0),2),p))
def chi_test(name,ev_cat,bg_cat,levels):
    sh=np.array([(bg_cat==l).mean() for l in levels])
    for lab,mask in (("discovery",~split),("holdout",split)):
        obs=np.array([(ev_cat[mask]==l).sum() for l in levels]); n=obs.sum()
        c,p=stats.chisquare(obs,sh*n); big=levels[int(np.argmax(obs/(sh*n)))]
        rows.append((name,lab,n,"/".join(map(str,obs)),"/".join(f"{x:.0f}" for x in sh*n),f"max ratio {np.max(obs/(sh*n)):.2f} ({big})",p))
M4=["Anila","Anala","Indra","Varuna"]
for fr,F,B in (("sid",Fs,Bs),("trop",Ft,Bt)):
    chi_test(f"R1 mandala [{fr}]",F.mand.values,B.mand.values,M4)
    binom_test(f"R2 Moon sharp group [{fr}]",F.sharp.values,B.sharp.values)
    binom_test(f"R8 Jupiter=Saturn nakshatra [{fr}]",F.jupsat.values,B.jupsat.values)
    chi_test(f"R11 PT season [{fr}]",F.season.values,B.season.values,list(range(6)))
binom_test("R3 Moon in Rikta tithi",Fs.rikta.values,Bs.rikta.values)
binom_test("R4 within +/-1 tithi of new/full moon",Fs.syz.values,Bs.syz.values)
binom_test("R5 eclipse within +/-15 d",Qd<=15,Bd<=15); binom_test("R6 eclipse within +/-7 d",Qd<=7,Bd<=7)
for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: binom_test(f"R7 {p} retrograde",Fs[p+"_R"].values,Bs[p+"_R"].values)
binom_test("R9 planet pair within 3 deg",Fs.war3.values,Bs.war3.values)
binom_test("R10 Moon within 5 deg of planet/Sun",Fs.moonpl5.values,Bs.moonpl5.values)
for fr,F in (("sid",Fs),("trop",Ft)):
    for lab,mask in (("discovery",~split),("holdout",split)):
        for var in ("mag","depth"):
            g=[Q.loc[mask&(F.mand.values==m),var].values for m in M4]; h,p=stats.kruskal(*g)
            rows.append((f"R12 mandala vs {var} [{fr}]",lab,int(mask.sum()),f"H={h:.2f}","",  "",p))
R=pd.DataFrame(rows,columns=["test","set","n","observed","expected","ratio_or_note","p"])
H=R[R.set=="holdout"]; m=23; R["bonf_threshold"]=0.05/m
R["p"]=R.p.astype(float); R["significant_after_bonferroni"]=(R.p<0.05/m)
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60)
print(R.round(4).to_string())
R.to_csv("results/prereg_large_sample_results.csv",index=False)
Q[["id","time","latitude","longitude","depth","mag","place"]].assign(holdout=split).to_csv("results/declustered_events_M65.csv",index=False)
print("holdout min p:",R[R.set=="holdout"].p.min(),"n tests per set:",(R.set=="holdout").sum())
