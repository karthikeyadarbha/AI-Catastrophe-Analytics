import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import eclipses as E
W=360/27
Q=pd.read_csv("quakes_m55_input.csv"); Q["t"]=pd.to_datetime(Q.datetime).dt.tz_localize(None)
S=pd.read_csv("m55_features_sid.csv"); B=pd.read_csv("bg_features_myeph.csv"); bg=pd.read_csv("bg_sid.csv"); bt_=pd.to_datetime(bg.datetime).dt.tz_localize(None)
P7=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
def trop(df):
    d=df.copy()
    for p in P7: d[p+"_Lon"]=(d[p+"_Lon"]+d.ay)%360
    return d
St,Bt=trop(S),trop(B)
split=(Q.t>=pd.Timestamp("2008-01-01")).values; bsplit=(bt_>=pd.Timestamp("2008-01-01")).values
new=(Q.mag<6.5).values; print("events",len(Q),"new",new.sum(),"M6.5+",(~new).sum(),"disc/hold",(~split).sum(),split.sum())
MAND={}
for name,l in {"Anila":[14,22,0,11,12,13],"Anala":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]}.items():
    for k in l: MAND[k]=name
BSM={}
for name,l in {"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]}.items():
    for k in l: BSM[k]=name
SHARP=[5,8,17,18]
nk=lambda x:(x//W).astype(int)
# ---------------- PRIMARY
rows=[]
def bt(name,ev,bgv,mask,bmask=None):
    p0=(bgv if bmask is None else bgv[bmask]).mean(); n=int(mask.sum()); o=int(ev[mask].sum()); return (name,n,o,round(n*p0,1),round(o/(n*p0),3),stats.binom.sf(o-1,n,p0))
sharpE=np.isin(nk(S.Moon_Lon.values),SHARP); sharpB=np.isin(nk(B.Moon_Lon.values),SHARP)
visE=(S.Sun_Lon.values>=200)&(S.Sun_Lon.values<200+W*1.0); visB=(B.Sun_Lon.values>=200)&(B.Sun_Lon.values<200+W)
prim=[bt("P1 Moon in sharp group, NEW events (M5.5-6.49), all years",sharpE,sharpB,new),
      bt("P1 same, discovery half (<2008)",sharpE,sharpB,new&~split),bt("P1 same, holdout half (2008+)",sharpE,sharpB,new&split),
      bt("P2 Sun in Vishakha, NEW events, all years",visE,visB,new),
      bt("P2 same, discovery half",visE,visB,new&~split),bt("P2 same, holdout half",visE,visB,new&split),
      bt("(context) M6.5+ events: Moon sharp group",sharpE,sharpB,~new),bt("(context) M6.5+ events: Sun in Vishakha",visE,visB,~new),
      bt("(context) ALL events: Moon sharp group",sharpE,sharpB,np.ones(len(Q),bool)),bt("(context) ALL events: Sun in Vishakha",visE,visB,np.ones(len(Q),bool))]
PR=pd.DataFrame(prim,columns=["test","n","observed","expected","ratio","p_upper"]); pd.set_option("display.width",250); pd.set_option("display.max_colwidth",70); print(PR.round(4).to_string()); PR.to_csv("results/m55_primary_tests.csv",index=False)
# ---------------- S1: 23 rules on all M5.5+ events, matched baselines
SEAS=lambda sun,anchor:(((sun-anchor)%360)//60).astype(int)
def feats(df,frame):
    L={p:df[f"{p}_Lon"].values for p in P7}; f=pd.DataFrame(index=df.index)
    f["mand"]=[MAND[k] for k in nk(L["Moon"])]; f["sharp"]=np.isin(nk(L["Moon"]),SHARP)
    f["season"]=SEAS(L["Sun"],22*W if frame=="sid" else 270.0); f["jupsat"]=nk(L["Jupiter"])==nk(L["Saturn"])
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: f[p+"_R"]=df[p+"_Is_Retro"].astype(bool).values
    el=(L["Moon"]-L["Sun"])%360; tith=(el//12).astype(int)+1; t15=(tith-1)%15+1
    f["rikta"]=np.isin(t15,[4,9,14]); f["syz"]=np.isin(tith,[29,30,1,14,15,16])
    pl=np.vstack([L[p] for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]]).T; war=np.zeros(len(df),bool)
    for i in range(5):
        for j in range(i+1,5): war|=np.abs((pl[:,i]-pl[:,j]+180)%360-180)<=3
    f["war3"]=war; oth=np.vstack([L[p] for p in ["Sun","Mars","Mercury","Jupiter","Venus","Saturn"]]).T
    f["moonpl5"]=(np.abs((L["Moon"][:,None]-oth+180)%360-180)<=5).any(1); return f
ev=E.find(1989,2028); ed=np.array([E.dt_to_jd(e["date"]) for e in ev])
def ecl(dt): jd=2440587.5+(pd.to_datetime(dt)-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400; return np.abs(jd[:,None]-ed[None,:]).min(1)
Fs,Ft,Gs,Gt=feats(S,"sid"),feats(St,"trop"),feats(B,"sid"),feats(Bt,"trop")
Qd=ecl(Q.t); Bd=ecl(bt_)
rows=[]
def b1(name,ce,cb):
    for lab,m,bm in (("discovery",~split,~bsplit),("holdout",split,bsplit)):
        p0=cb[bm].mean(); n=int(m.sum()); o=int(ce[m].sum()); rows.append((name,lab,n,str(o),f"{n*p0:.1f}",round(o/(n*p0),3),stats.binom.sf(o-1,n,p0)))
def c1(name,ce,cb,lv):
    for lab,m,bm in (("discovery",~split,~bsplit),("holdout",split,bsplit)):
        sh=np.array([(cb[bm]==l).mean() for l in lv]); o=np.array([(ce[m]==l).sum() for l in lv]); n=o.sum(); rows.append((name,lab,n,"/".join(map(str,o)),"/".join(f"{x:.0f}" for x in sh*n),round(float(np.max(o/(sh*n))),3),stats.chisquare(o,sh*n)[1]))
for fr,F,G in (("sid",Fs,Gs),("trop",Ft,Gt)):
    c1(f"R1 mandala [{fr}]",F.mand.values,G.mand.values,["Anila","Anala","Indra","Varuna"]); b1(f"R2 Moon sharp group [{fr}]",F.sharp.values,G.sharp.values)
    b1(f"R8 Jupiter=Saturn nakshatra [{fr}]",F.jupsat.values,G.jupsat.values); c1(f"R11 PT season [{fr}]",F.season.values,G.season.values,list(range(6)))
b1("R3 Moon in Rikta tithi",Fs.rikta.values,Gs.rikta.values); b1("R4 within +/-1 tithi of new/full moon",Fs.syz.values,Gs.syz.values)
b1("R5 eclipse within +/-15 d",Qd<=15,Bd<=15); b1("R6 eclipse within +/-7 d",Qd<=7,Bd<=7)
for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: b1(f"R7 {p} retrograde",Fs[p+"_R"].values,Gs[p+"_R"].values)
b1("R9 planet pair within 3 deg",Fs.war3.values,Gs.war3.values); b1("R10 Moon within 5 deg of planet/Sun",Fs.moonpl5.values,Gs.moonpl5.values)
M4=["Anila","Anala","Indra","Varuna"]
for fr,F in (("sid",Fs),("trop",Ft)):
    for lab,m in (("discovery",~split),("holdout",split)):
        for var,vals in (("mag",Q.mag.values),("depth",Q.depth.values)):
            g=[vals[m&(F.mand.values==k)] for k in M4]; h,p=stats.kruskal(*g); rows.append((f"R12 mandala vs {var} [{fr}]",lab,int(m.sum()),f"H={h:.2f}","","",p))
R=pd.DataFrame(rows,columns=["test","set","n","observed","expected","ratio","p"]); R["p"]=R.p.astype(float); R["threshold"]=0.05/23; R["significant"]=R.p<0.05/23
print(R[R.set=="holdout"].round(4).to_string()); print("min holdout p",R[R.set=="holdout"].p.min().round(5),"min disc p",R[R.set=="discovery"].p.min().round(5)); R.to_csv("results/m55_23rules_results.csv",index=False)
# ---------------- S3 mandalas on NEW events
for nm,MM,lv in (("PT",MAND,M4),("BS",BSM,["Vayu","Agni","Indra","Varuna"])):
    for fr,df,bdf in (("sid",S,B),("trop",St,Bt)):
        mq=np.array([MM[k] for k in nk(df.Moon_Lon.values[new])]); mb=np.array([MM[k] for k in nk(bdf.Moon_Lon.values)])
        sh=np.array([(mb==l).mean() for l in lv]); o=np.array([(mq==l).sum() for l in lv]); print("S3",nm,fr,o.tolist(),np.round(sh*o.sum(),0).tolist(),"p",round(stats.chisquare(o,sh*o.sum())[1],4))

bands=[("5.5-5.9",(Q.mag>=5.5)&(Q.mag<6.0)),("6.0-6.4",(Q.mag>=6.0)&(Q.mag<6.5)),("6.5-6.9",(Q.mag>=6.5)&(Q.mag<7.0)),("7.0+",Q.mag>=7.0)]
oth=lambda df,ps:np.vstack([df[f"{p}_Lon"].values for p in ps]).T
def near(df,ps): return (np.abs((df.Moon_Lon.values[:,None]-oth(df,ps)+180)%360-180)<=5).any(1)
allp=["Sun","Mars","Mercury","Jupiter","Venus","Saturn"]
feat={"sharp group":(sharpE,sharpB),"Sun in Vishakha":(visE,visB),"Moon within 5deg of Sun/planet":(near(S,allp),near(B,allp)),"Moon within 5deg of Sun only":(near(S,["Sun"]),near(B,["Sun"])),"Moon within 5deg of a planet (not Sun)":(near(S,allp[1:]),near(B,allp[1:]))}
rows=[]
for fn,(e,b) in feat.items():
    p0=b.mean()
    for bn,m in bands:
        m=m.values; n=int(m.sum()); o=int(e[m].sum()); rows.append((fn,bn,n,o,round(n*p0,1),round(o/(n*p0),3),stats.binom.sf(o-1,n,p0)))
for p in allp[1:]:
    e,b=near(S,[p]),near(B,[p]); p0=b.mean(); m=new; n=int(m.sum()); o=int(e[m].sum()); rows.append((f"Moon within 5deg of {p} (new events)","M5.5-6.49",n,o,round(n*p0,1),round(o/(n*p0),3),stats.binom.sf(o-1,n,p0)))
Bd=pd.DataFrame(rows,columns=["feature","band","n","observed","expected","ratio","p_upper"]); print(Bd.round(4).to_string()); Bd.to_csv("results/m55_by_magnitude_band.csv",index=False)
