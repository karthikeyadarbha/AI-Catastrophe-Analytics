import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
sys.path.insert(0,"../shared"); import myeph as M
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
d=pd.read_csv("train.csv"); d['t']=pd.to_datetime(d.time,format='ISO8601',utc=True).dt.tz_localize(None)
jd=lambda t:2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
d['jd']=jd(d.t)
def feats(J):
    A=M.all_trop(J); ay=M.ayan(J); o={}
    for p in ["Sun","Moon","Mars","Jupiter","Saturn","Mercury","Venus"]: o[p]=(A[p]-ay)%360
    o["Node"]=M.node_trop(J)
    return o
r=np.radians
def dec(lon_trop,lat): e=r(23.439); return np.degrees(np.arcsin(np.sin(r(lat))*np.cos(e)+np.cos(r(lat))*np.sin(e)*np.sin(r(lon_trop))))
# 1. validate file against own ephemeris (1850+)
m=d.t.dt.year>=1850; F=feats(d.jd.values[m])
print("validation (events 1850+):",m.sum())
for p,c in [("Moon","Sid_Topo_Moon_Lon"),("Sun","Sid_Topo_Sun_Lon"),("Mars","Sid_Topo_Mars_Lon"),("Jupiter","Sid_Topo_Jup_Lon"),("Saturn","Sid_Topo_Sat_Lon")]:
    x=((F[p]-d[c].values[m]+180)%360)-180; print(p,"median|diff|",round(np.median(abs(x)),2),"p90",round(np.quantile(abs(x),.9),2),"max",round(abs(x).max(),2))
x=((F["Node"]-M.ayan(d.jd.values[m])-d.Topo_Node_Lon.values[m]+180)%360)-180; print("Node median|diff|",round(np.median(abs(x)),2))
# Phase angle: file vs Moon-Sun elongation
el=((F["Moon"]-F["Sun"])%360); pa=d.Phase_Angle.values[m]; print("Phase_Angle vs elongation:",round(np.median(np.minimum(abs(el-pa),360-abs(el-pa))),2),"; vs 180-|el-180|:",round(np.median(abs((180-abs(el-180))-pa)),2),"; median file",np.median(pa))
# 2. rule tests on 1900+ events, uniform background 1900-2025
e=d[d.t.dt.year>=1900].copy(); n=len(e); print("events used",n)
rng=np.random.default_rng(1); NB=200000
Jb=rng.uniform(e.jd.min(),e.jd.max(),NB); B=feats(Jb)
E=feats(e.jd.values); 
nk=lambda L:(L//(360/27)).astype(int)
sharp=[5,8,17,18]; 
PT={"Anila":[14,22,0,11,12,13],"Anala":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]}
BSm={"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]}
def binom(name,ev,bg):
    p0=bg.mean(); k=int(ev.sum()); pv=stats.binomtest(k,n,p0,alternative="greater").pvalue; print(f"{name:42s} {k:3d} vs {p0*n:6.1f} ({k/(p0*n):.2f}x) p={pv:.3f}")
    return pv
res=[]
res.append(binom("Moon in sharp group",np.isin(nk(E["Moon"]),sharp),np.isin(nk(B["Moon"]),sharp)))
res.append(binom("Sun in Visakha",nk(E["Sun"])==15,nk(B["Sun"])==15))
sep=lambda F_:np.min([abs(((F_["Moon"]-F_[p]+180)%360)-180) for p in ["Sun","Mars","Mercury","Jupiter","Venus","Saturn"]],axis=0)
res.append(binom("Moon within 5 deg of Sun/planet",sep(E)<=5,sep(B)<=5))
syz=lambda F_:(lambda el:np.minimum(el,360-el))((F_["Moon"]-F_["Sun"])%360)
res.append(binom("Moon within 12 deg of new/full (syzygy)",(lambda x:np.minimum(x,180-x)<=12)(syz(E)),(lambda x:np.minimum(x,180-x)<=12)(syz(B))))
def chi(name,dic):
    names=list(dic); cls=lambda L:np.array([next((i for i,k in enumerate(names) if x in dic[k]),-1) for x in nk(L)])
    ce=np.bincount(cls(E["Moon"]),minlength=4)[:4]; cb=np.bincount(cls(B["Moon"]),minlength=4)[:4]; ex=cb/cb.sum()*n
    c,p=stats.chisquare(ce,ex*ce.sum()/ex.sum()); print(f"{name:42s} {ce.tolist()} vs {np.round(ex,1).tolist()} p={p:.3f}"); return p
res.append(chi("PT mandala by Moon nakshatra",PT)); res.append(chi("BS mandala by Moon nakshatra",BSm))
print("Bonferroni 0.05/6 =",round(0.05/6,4),"min p",round(min(res),3))
# 3. model: file's features vs matched negatives
def X(F_,lat):
    L=F_["Moon"]; el=(F_["Moon"]-F_["Sun"])%360
    cols=[np.abs(dec(F_["Moon"]+M.ayan(0)*0+0,lat))]  # placeholder replaced below
    return None
def design(F_,J,lat):
    ay=M.ayan(J); mt=(F_["Moon"]+ay)%360
    el=(F_["Moon"]-F_["Sun"])%360; ph=np.minimum(el,360-el)
    out=[np.abs(dec(mt,np.zeros_like(mt))*0+np.degrees(np.arcsin(np.sin(r(23.439))*np.sin(r(mt))))), ph]
    for p in ["Moon","Sun","Mars","Jupiter","Saturn"]: out+= [np.sin(r(F_[p])),np.cos(r(F_[p]))]
    out+=[np.sin(r(F_["Node"]-ay)),np.cos(r(F_["Node"]-ay))]
    return np.c_[tuple(out)]
Xp=design(E,e.jd.values,e.latitude.values)
auc=[]
for rep in range(10):
    Jn=rng.uniform(e.jd.min(),e.jd.max(),n*20); Fn=feats(Jn); Xn=design(Fn,Jn,None)
    Xa=np.r_[Xp,Xn]; ya=np.r_[np.ones(n),np.zeros(len(Xn))]
    p=cross_val_predict(RandomForestClassifier(300,min_samples_leaf=3,class_weight="balanced",random_state=rep,n_jobs=-1),Xa,ya,cv=StratifiedKFold(5,shuffle=True,random_state=rep),method="predict_proba")[:,1]
    auc.append(roc_auc_score(ya,p))
print("RF CV AUC over 10 repeats: mean",round(np.mean(auc),3),"min",round(min(auc),3),"max",round(max(auc),3))
sh=[]
for rep in range(10):
    ys=rng.permutation(ya); p=cross_val_predict(RandomForestClassifier(300,min_samples_leaf=3,class_weight="balanced",random_state=rep,n_jobs=-1),Xa,ys,cv=StratifiedKFold(5,shuffle=True,random_state=rep),method="predict_proba")[:,1]; sh.append(roc_auc_score(ys,p))
print("shuffled-label AUC mean",round(np.mean(sh),3),"max",round(max(sh),3))
