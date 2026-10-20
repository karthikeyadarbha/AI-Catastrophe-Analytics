import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd,itertools
from scipy import stats
sys.path.insert(0,"."); import myeph
sys.path.insert(0,"../shared"); import eclipses as E
W=360/27; P5=["Mars","Mercury","Jupiter","Venus","Saturn"]; P7=["Sun","Moon"]+P5
Q=pd.read_csv("quakes_m55_input.csv"); Q["t"]=pd.to_datetime(Q.datetime).dt.tz_localize(None)
bgd=pd.read_csv("bg_sid.csv")[["datetime"]]; bgd["t"]=pd.to_datetime(bgd.datetime).dt.tz_localize(None)
def build(df):
    jd=2440587.5+(df.t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400; A=myeph.all_trop(jd); ay=myeph.ayan(jd)
    D={"ay":ay,"jd":jd}
    for p in P7: D[p]=A[p]%360
    for p in P5: D[p+"_retro"]=A[p+"_retro"]; D[p+"_spd"]=myeph.speed(p,jd)
    D["Rahu"]=myeph.node_trop(jd); return D
DQ,DB=build(Q),build(bgd)
split=(Q.t>=pd.Timestamp("2008-01-01")).values; bsplit=(bgd.t>=pd.Timestamp("2008-01-01")).values
nk=lambda x:(x//W).astype(int)
def sid(D,p): return (D[p]-D["ay"])%360
def trp(D,p): return D[p]%360
ev=E.find(1989,2028); ed=np.array([E.dt_to_jd(e["date"]) for e in ev]); ecl=lambda jd:np.abs(jd[:,None]-ed[None,:]).min(1)
sep=lambda a,b:np.abs((a-b+180)%360-180)
def flags(D):
    F={}
    L={p:sid(D,p) for p in P7}; Lt={p:trp(D,p) for p in P7}; nr=(L["Mars"]*0)
    # G1
    for fr,LL in (("sid",L),("trop",Lt)):
        n=np.vstack([nk(LL[p]) for p in P5]).T; F[f"G1 formation 3+ planets in one nakshatra [{fr}]"]=np.array([np.bincount(r,minlength=27).max()>=3 for r in n])
        VEN={"Adya":range(1,5),"Rohita":range(5,9),"Daruna":range(9,13),"Virocana":range(13,17),"Urdhvadanda":range(17,22),"Tiksna":range(22,27)}
        vn=nk(LL["Venus"]); lab=np.array(["none"]*len(vn),dtype=object)
        for k,v in VEN.items(): lab[np.isin(vn,list(v))]=k
        F[f"G2 Venus circle [{fr}]"]=lab
    F["G3 Mars stationary"]=np.abs(D["Mars_spd"])<0.05
    F["G3 Saturn in Krittika & Jupiter in Vishakha"]=(nk(L["Saturn"])==2)&(nk(L["Jupiter"])==15)
    for nm,ang in (("opposition",180),("square",90),("trine",120)):
        w=np.zeros(len(L["Sun"]),bool)
        for a,b in itertools.combinations(P5,2): w|=np.abs(sep(L[a],L[b])-ang)<=3
        F[f"G4 any planet pair within 3 deg of {nm}"]=w
    F["G5 Rahu nakshatra"]=nk(sid(D,"Rahu")); ra=sid(D,"Rahu"); axis=np.minimum(sep(L["Moon"],ra),sep(L["Moon"],(ra+180)%360)); F["G5 Moon within 12 deg of node axis"]=axis<=12
    mn=nk(L["Moon"]); F["G6 Moon in same nakshatra as any of 5 planets"]=np.any([mn==nk(L[p]) for p in P5],axis=0); mal=np.any([mn==nk(L[p]) for p in ("Mars","Saturn")],axis=0); F["G6 Moon in same nakshatra as Mars or Saturn"]=mal
    ORB={"Mars":17,"Mercury":14,"Jupiter":11,"Venus":10,"Saturn":15}; comb={}
    for p,o in ORB.items(): comb[p]=sep(L[p],L["Sun"])<=o; F[f"G7 {p} combust"]=comb[p]
    F["G7 two or more planets combust"]=np.sum([comb[p] for p in P5],axis=0)>=2
    # base conditions for score and screen
    el=(L["Moon"]-L["Sun"])%360; tith=(el//12).astype(int)+1; t15=(tith-1)%15+1
    base={"Moon sharp group":np.isin(mn,[5,8,17,18]),"Rikta tithi":np.isin(t15,[4,9,14]),"new/full +-1 tithi":np.isin(tith,[29,30,1,14,15,16]),"eclipse +-15d":ecl(D["jd"])<=15,
      "Mars retro":D["Mars_retro"],"Saturn retro":D["Saturn_retro"],"Jupiter retro":D["Jupiter_retro"],"Mercury retro":D["Mercury_retro"],"Venus retro":D["Venus_retro"]}
    w=np.zeros(len(mn),bool)
    for a,b in itertools.combinations(P5,2): w|=sep(L[a],L[b])<=3
    base["planet pair <=3deg"]=w; base["Moon <=5deg of planet/Sun"]=np.any([sep(L["Moon"],L[p])<=5 for p in ["Sun"]+P5],axis=0)
    base["any planet combust"]=np.any([comb[p] for p in P5],axis=0); base["Moon near node axis"]=axis<=12; base["Moon same nakshatra as Mars/Saturn"]=mal
    F["_base"]=base
    mal5=np.any([sep(L["Moon"],L[p])<=5 for p in ("Mars","Saturn")],axis=0)
    F["G9 affliction score"]=(base["Moon sharp group"].astype(int)+base["Rikta tithi"]+base["Mars retro"]+base["Saturn retro"]+mal5+comb["Mars"]+comb["Saturn"]+base["Moon near node axis"]+mal).astype(float)
    return F
FQ,FB=flags(DQ),flags(DB)

import warnings; warnings.filterwarnings("ignore")
def stat_set(jd):
    A=myeph.all_trop(jd); ay=myeph.ayan(jd); L={p:(A[p]-ay)%360 for p in P7}; Lt={p:A[p]%360 for p in P7}
    ra=(myeph.node_trop(jd)-ay)%360; rn=nk(ra)
    out={"rahu_chi":np.bincount(rn,minlength=27)}
    n=np.vstack([nk(L[p]) for p in P5]).T; out["form_sid"]=np.array([np.bincount(r,minlength=27).max()>=3 for r in n]).sum()
    n=np.vstack([nk(Lt[p]) for p in P5]).T; out["form_trop"]=np.array([np.bincount(r,minlength=27).max()>=3 for r in n]).sum()
    out["mars_stat"]=(np.abs(myeph.speed("Mars",jd))<0.05).sum()
    return out
jdq=DQ["jd"]; obs=stat_set(jdq); jdb=DB["jd"]; bgr=stat_set(jdb); pr=bgr["rahu_chi"]/bgr["rahu_chi"].sum()
chi=lambda c:((c-pr*c.sum())**2/(pr*c.sum())).sum()
rng=np.random.default_rng(8); nul={"rahu":[],"form_sid":[],"form_trop":[],"mars_stat":[]}
for _ in range(400):
    off=rng.uniform(400,5000)*rng.choice([-1,1]); o=stat_set(jdq+off); nul["rahu"].append(chi(o["rahu_chi"])); nul["form_sid"].append(o["form_sid"]); nul["form_trop"].append(o["form_trop"]); nul["mars_stat"].append(o["mars_stat"])
res=[]
for k,ov in (("rahu",chi(obs["rahu_chi"])),("form_sid",obs["form_sid"]),("form_trop",obs["form_trop"]),("mars_stat",obs["mars_stat"])):
    x=np.array(nul[k]); p=(np.sum(x>=ov)+1)/401; res.append((k,float(ov),float(x.mean()),float(x.std()),p)); print(k,"obs",round(ov,1),"shift-null mean",round(x.mean(),1),"sd",round(x.std(),1),"p",round(p,4))
pd.DataFrame(res,columns=["test","observed","shift_null_mean","shift_null_sd","p_shift"]).to_csv("results/gap_slow_planet_time_shift_check.csv",index=False)
