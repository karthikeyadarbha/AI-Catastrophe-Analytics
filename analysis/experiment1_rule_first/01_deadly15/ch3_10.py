import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import numpy as np, pandas as pd
from scipy import stats
exec(open(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),"ch14_16.py")).read().split("q=pd.read_csv")[0])   # reuse D (sidereal Lahiri longitudes)
q=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv").set_index("Event")
q["yr"]=pd.to_datetime(q.UTC).dt.year+pd.to_datetime(q.UTC).dt.dayofyear/365.25
RETRO={"Sumatra 2004":["Sat"],"Haiti 2010":["Mars","Mer"],"Sichuan 2008":["Jup"],"Kashmir 2005":["Mars"],"Turkey-Syria 2023":[],"Bam 2003":["Mer","Sat"],"Bhuj 2001":[],"Tohoku 2011":["Sat"],"Izmit 1999":["Ven"],"Nepal 2015":["Sat"],"Venezuela 2026":[],"Java 2006":["Jup"],"Sulawesi 2018":[],"Myanmar 2025":["Mer","Ven"],"Morocco 2023":["Mer","Jup","Sat"]}
COMB={"Sumatra 2004":0,"Haiti 2010":1,"Sichuan 2008":1,"Kashmir 2005":0,"Turkey-Syria 2023":1,"Bam 2003":1,"Bhuj 2001":0,"Tohoku 2011":2,"Izmit 1999":1,"Nepal 2015":1,"Venezuela 2026":0,"Java 2006":1,"Sulawesi 2018":1,"Myanmar 2025":2,"Morocco 2023":1}
W=360/27
VEN={"Adya":[1,2,3,4],"Rohita":[5,6,7,8],"Daruna":[9,10,11,12],"Virocana":[13,14,15,16],"Urdhvadanda":[17,18,19,20,21],"Tiksna":[22,23,24,25,26]}
GRP={"Dhruva":[3,11,20,25],"Mridu":[4,13,16,26],"Ugra":[9,1,10,19,24],"Tikshna":[5,8,17,18],"Kshipra":[7,0,12],"Chara":[6,14,21,22,23],"Mixed":[2,15]}
SEAS=["Sisira","Vasanta","Grisma","Varsa","Sarat","Hemanta"]
rows=[]
for ev,L in D.items():
    L=np.array(L); ay=23.853+(q.loc[ev,"yr"]-2000)*0.013972
    r=dict(Event=ev)
    for frame,off in (("sid",0.0),("trop",ay)):
        P=(L+off)%360
        sun,moon,mars,mer,jup,ven,sat=P
        nk=lambda x:int(x//W)
        r[f"Moon_nak_{frame}"]=nk(moon)
        r[f"Venus_mandala_{frame}"]=next((k for k,v in VEN.items() if nk(ven) in v),"none")
        r[f"Moon_group_{frame}"]=next(k for k,v in GRP.items() if nk(moon) in v)
        anchor=22*W if frame=="sid" else 270.0
        r[f"PT_season_{frame}"]=SEAS[int(((sun-anchor)%360)//60)]
        r[f"JupSat_same_nak_{frame}"]=int(nk(jup)==nk(sat))
        r[f"Sat_Krittika_Jup_Vishakha_{frame}"]=int(nk(sat)==2 and nk(jup)==15)
        pl=[mars,mer,jup,ven,sat]; r[f"max_cluster_{frame}"]=int(np.bincount([nk(x) for x in pl],minlength=27).max())
        r[f"Mars_nak_{frame}"]=nk(mars)
    r["n_retro"]=len(RETRO[ev]); r["n_combust"]=COMB[ev]
    for p in ["Mars","Mer","Jup","Ven","Sat"]: r[f"{p}_retro"]=int(p in RETRO[ev])
    rows.append(r)
F=pd.DataFrame(rows).set_index("Event").join(q[["Deaths","Mag"]]); F["logD"]=np.log10(F.Deaths)
print(F[[c for c in F.columns if "_sid" in c or "_trop" in c][:0]+["Moon_group_sid","Moon_group_trop","Venus_mandala_sid","Venus_mandala_trop","PT_season_sid","PT_season_trop","JupSat_same_nak_sid","JupSat_same_nak_trop","max_cluster_sid","max_cluster_trop","n_retro","n_combust"]].to_string())
res=[]
def add(name,o,p):
    res.append((name,int(o),round(15*p,2),round(stats.binom.sf(o-1,15,p),3)))
for fr in ("sid","trop"):
    for k,v in VEN.items(): add(f"[{fr}] Venus in {k} mandala",(F[f"Venus_mandala_{fr}"]==k).sum(),len(v)/27)
    for k,v in GRP.items(): add(f"[{fr}] Moon nak group {k}",(F[f"Moon_group_{fr}"]==k).sum(),len(v)/27)
    for s in SEAS: add(f"[{fr}] PT season {s}",(F[f"PT_season_{fr}"]==s).sum(),1/6)
    add(f"[{fr}] Jupiter & Saturn same nakshatra",F[f"JupSat_same_nak_{fr}"].sum(),1/27)
    add(f"[{fr}] >=3 of 5 planets one nakshatra",(F[f"max_cluster_{fr}"]>=3).sum(),0.014)
for p,pr in zip(["Mars","Mer","Jup","Ven","Sat"],[.093,.19,.30,.07,.36]): add(f"{p} retrograde",F[f"{p}_retro"].sum(),pr)
R=pd.DataFrame(res,columns=["test","observed_of_15","expected","p_upper"]); R["p_bonf"]=(R.p_upper*len(R)).clip(upper=1).round(3)
print(R.to_string()); print("n tests",len(R))
# ---- correlations with severity
C=[]
for c in ["n_retro","n_combust","max_cluster_sid","max_cluster_trop","JupSat_same_nak_sid","JupSat_same_nak_trop","Moon_nak_sid","Moon_nak_trop","Mars_nak_sid","Mars_nak_trop"]:
    for t in ("logD","Mag"):
        rho,p=stats.spearmanr(F[c],F[t]); C.append((c,t,round(rho,2),round(p,3)))
print(pd.DataFrame(C,columns=["feature","vs","spearman","p"]).to_string())
# categorical vs severity: Kruskal
K=[]
for c in ["Venus_mandala_sid","Venus_mandala_trop","Moon_group_sid","Moon_group_trop","PT_season_sid","PT_season_trop"]:
    for t in ("logD","Mag"):
        g=[x[t].values for _,x in F.groupby(c) if len(x)>=1]
        if len(g)>1: h,p=stats.kruskal(*g); K.append((c,t,len(g),round(h,2),round(p,3)))
print(pd.DataFrame(K,columns=["category","vs","groups","H","p"]).to_string())
F.to_csv("results/ch3_10_features_sidereal_tropical.csv"); R.to_csv("results/ch3_10_tests.csv",index=False)
