import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import myeph as M
from hist import *
era=sys.argv[1]   # A or B
if era=="A":
    a=pd.read_csv("histA_m55_65.csv"); b=pd.read_csv("old_m65.csv"); q=pd.concat([a,b]).drop_duplicates("id"); y0,y1=1964,1989
else:
    q=pd.read_csv("histB_m7.csv"); y0,y1=1900,1963
q=q[(q.type=="earthquake")&(q.mag>=(5.5 if era=="A" else 7.0))].copy(); q["tt"]=pd.to_datetime(q.time).dt.tz_localize(None); q=q[(q.tt.dt.year>=y0)&(q.tt.dt.year<=y1)]
print("raw",len(q)); Q=decluster(q); print("declustered",len(Q),"by mag",Q.mag.describe().round(2).to_dict())
Q[["id","datetime","latitude","longitude","mag","depth"]].to_csv(f"hist{era}_declustered.csv",index=False)
S=M.enrich(Q); G=grid(y0,y1); B=M.enrich(G); ayS=M.ayan(2440587.5+(pd.to_datetime(Q.datetime).dt.tz_localize(None)-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400); ayB=M.ayan(2440587.5+(G.datetime-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400)
St,Bt=trop(S,ayS),trop(B,ayB); print("events",len(S),"background",len(B))
nk=lambda x:(x//W).astype(int)
MAND={}
for name,l in {"Anila":[14,22,0,11,12,13],"Anala":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]}.items():
    for k in l: MAND[k]=name
SHARP=[5,8,17,18]; mag=Q.mag.values; dep=Q.depth.values
oth=lambda df:np.vstack([df[f"{p}_Lon"].values for p in ["Sun","Mars","Mercury","Jupiter","Venus","Saturn"]]).T
def r10(df): return (np.abs((df.Moon_Lon.values[:,None]-oth(df)+180)%360-180)<=5).any(1)
eE,eB=r10(S),r10(B); p0=eB.mean(); print("background rate R10",round(p0,4))
def bt(name,ev,bgv,mask):
    n=int(mask.sum()); o=int(ev[mask].sum()); e=n*bgv.mean(); return (name,n,o,round(e,1),round(o/e,3),stats.binom.sf(o-1,n,bgv.mean()))
rows=[]
if era=="A":
    rows+=[bt("PRIMARY H1: Moon within 5 deg of Sun/planet, M5.5-5.9",eE,eB,(mag<6.0)),bt("S1 all M>=5.5",eE,eB,mag>=5.5),bt("S1 M>=6.0",eE,eB,mag>=6.0)]
else: rows+=[bt("S2 Era B M>=7.0",eE,eB,np.ones(len(Q),bool))]
# time-shift null for R10 on the primary subset
sub=(mag<6.0) if era=="A" else np.ones(len(Q),bool)
tq=pd.to_datetime(Q.datetime).dt.tz_localize(None); rng=np.random.default_rng(11); obs=eE[sub].mean(); nulls=[]
for _ in range(500):
    sh=int(rng.integers(-3650,3651)); t2=(tq[sub]+pd.Timedelta(days=sh)).reset_index(drop=True); d=pd.DataFrame({"datetime":t2}); nulls.append(r10(M.enrich(d)).mean())
nulls=np.array(nulls); tsp=(nulls>=obs).mean(); print("time-shift: obs rate",round(obs,4),"null mean",round(nulls.mean(),4),"sd",round(nulls.std(),4),"p",tsp)
rows.append(("S5 time-shift p for H1 subset",int(sub.sum()),"","",round(obs/nulls.mean(),3),tsp))
# S3 / 23-rule family (21 rules without eclipses)
fam=[]
def b1(name,ce,cb): n=len(ce); o=int(ce.sum()); e=n*cb.mean(); fam.append((name,n,str(o),f"{e:.1f}",round(o/e,3),stats.binom.sf(o-1,n,cb.mean())))
def c1(name,ce,cb,lv):
    sh=np.array([(cb==l).mean() for l in lv]); o=np.array([(ce==l).sum() for l in lv]); n=o.sum(); fam.append((name,n,"/".join(map(str,o)),"/".join(f"{x:.0f}" for x in sh*n),round(float(np.max(o/(sh*n))),3),stats.chisquare(o,sh*n)[1]))
SEAS=lambda sun,anchor:(((sun-anchor)%360)//60).astype(int)
def feats(df,frame):
    L={p:df[f"{p}_Lon"].values for p in P7}; f=pd.DataFrame(index=range(len(df)))
    f["mand"]=[MAND[k] for k in nk(L["Moon"])]; f["sharp"]=np.isin(nk(L["Moon"]),SHARP)
    f["season"]=SEAS(L["Sun"],22*W if frame=="sid" else 270.0); f["jupsat"]=nk(L["Jupiter"])==nk(L["Saturn"]); f["vish"]=(L["Sun"]>=200)&(L["Sun"]<200+W)
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: f[p+"_R"]=df[p+"_Is_Retro"].astype(bool).values
    el=(L["Moon"]-L["Sun"])%360; tith=(el//12).astype(int)+1; t15=(tith-1)%15+1
    f["rikta"]=np.isin(t15,[4,9,14]); f["syz"]=np.isin(tith,[29,30,1,14,15,16])
    pl=np.vstack([L[p] for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]]).T; war=np.zeros(len(df),bool)
    for i in range(5):
        for j in range(i+1,5): war|=np.abs((pl[:,i]-pl[:,j]+180)%360-180)<=3
    f["war3"]=war; f["moonpl5"]=r10(df); return f
Fs,Ft,Gs,Gt=feats(S,"sid"),feats(St,"trop"),feats(B,"sid"),feats(Bt,"trop")
for fr,F,G in (("sid",Fs,Gs),("trop",Ft,Gt)):
    c1(f"R1 mandala [{fr}]",F.mand.values,G.mand.values,["Anila","Anala","Indra","Varuna"]); b1(f"R2 Moon sharp group [{fr}]",F.sharp.values,G.sharp.values)
    b1(f"R8 Jupiter=Saturn nakshatra [{fr}]",F.jupsat.values,G.jupsat.values); c1(f"R11 PT season [{fr}]",F.season.values,G.season.values,list(range(6)))
b1("R3 Moon in Rikta tithi",Fs.rikta.values,Gs.rikta.values); b1("R4 within +/-1 tithi of new/full moon",Fs.syz.values,Gs.syz.values)
for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: b1(f"R7 {p} retrograde",Fs[p+"_R"].values,Gs[p+"_R"].values)
b1("R9 planet pair within 3 deg",Fs.war3.values,Gs.war3.values); b1("R10 Moon within 5 deg of planet/Sun",Fs.moonpl5.values,Gs.moonpl5.values)
M4=["Anila","Anala","Indra","Varuna"]
for fr,F in (("sid",Fs),("trop",Ft)):
    for var,vals in (("mag",mag),("depth",dep)):
        g=[vals[F.mand.values==k] for k in M4]; h,p=stats.kruskal(*g); fam.append((f"R12 mandala vs {var} [{fr}]",len(Q),f"H={h:.2f}","","",p))
nF=len(fam); print("family size",nF)
b1("S4 Sun in Vishakha",Fs.vish.values,Gs.vish.values)
R=pd.DataFrame(rows+fam,columns=["test","n","observed","expected","ratio","p"]); R["p"]=R.p.astype(float); R["era"]=era
R["threshold"]=[0.05]*len(rows)+[0.05/21]*21+[0.025]; R["passes"]=(R.p<R.threshold)&(pd.to_numeric(R.ratio,errors="coerce").fillna(2)>1)
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60); print(R.round(4).to_string()); R.to_csv(f"hist_results_{era}.csv",index=False)
