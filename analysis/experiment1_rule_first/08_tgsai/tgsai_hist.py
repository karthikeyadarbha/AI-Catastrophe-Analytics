import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
sys.path.insert(0,"../shared"); import myeph as M
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from hist import decluster,grid
cols=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
def jd(ts): return (ts.astype("datetime64[s]").astype(np.int64))/86400.0+2440587.5
r=np.radians
def repo_feats(J,lat,kind,P=None):
    T=(J-2451545.0)/36525.0
    sun=(280.466+36000.77*T+1.915*np.sin(r(357.529+35999.05*T)))%360; sdec=np.degrees(np.arcsin(np.sin(r(23.439))*np.sin(r(sun))))
    Mm=(134.963+477198.8676*T)%360; moon=(218.316+481267.88*T+6.289*np.sin(r(Mm)))%360; mlat=5.128*np.sin(r(93.27+483202.01*T))
    mdec=np.degrees(np.arcsin(np.sin(r(mlat))*np.cos(r(23.439))+np.cos(r(mlat))*np.sin(r(23.439))*np.sin(r(moon))))
    if kind=="crude":
        pl=lambda Lm,pi,e:(Lm+np.degrees(2*e*np.sin(r(Lm-pi))))%360
        sat=pl(50.08+1222.1*T,92.06,0.0555); jup=pl(34.35+3034.9*T,14.33,0.0485); mars=pl(355.43+19140.3*T,336.06,0.0934); sunL=sun
    else: sat,jup,mars,sunL,moon=P["Saturn"],P["Jupiter"],P["Mars"],P["Sun"],P["Moon"]
    al=lambda a,b:(1+np.cos(r(2*(a-b))))/2
    return np.c_[al(moon,sunL),al(sat,sunL),al(jup,sunL),al(mars,sunL),(1+np.cos(r(Mm)))/2,np.cos(r(lat-mdec)),np.cos(r(lat-sdec))]
def PT(Ps):
    W=360/27; nk=(Ps["Moon"]//W).astype(int); el=(Ps["Moon"]-Ps["Sun"])%360; oh=np.eye(27)[nk]
    return np.c_[np.sin(r(Ps["Moon"])),np.cos(r(Ps["Moon"])),np.sin(r(el)),np.cos(r(el)),oh[:,[14,22,0,11,12,13]].sum(1),oh[:,[2,15,24,1,19,9,7,10]].sum(1),oh[:,[17,20,21,3,4,6,16]].sum(1),oh[:,[23,25,26,5,8,18]].sum(1)]
def build(t,lat,sid,ay):
    J=jd(pd.to_datetime(t).values if not hasattr(t,"values") else t.values)
    tro={c:(sid[c+"_Lon"].values+ay)%360 for c in cols}; sd={c:sid[c+"_Lon"].values for c in cols}
    A=repo_feats(J,lat,"crude"); B=repo_feats(J,lat,"true",tro); C=np.c_[B,PT(sd)]; return {"A":A,"B":B,"C":C}
# ---------------- training set 1990-2026
Q=pd.read_csv("quakes_m55_input.csv"); S=pd.read_csv("m55_features_sid.csv"); bg=pd.read_csv("bg_sid.csv"); Bf=pd.read_csv("bg_features_myeph.csv")
tq=pd.to_datetime(Q.datetime).dt.tz_localize(None); tb=pd.to_datetime(bg.datetime).dt.tz_localize(None)
rng=np.random.default_rng(5); idx=rng.integers(0,len(Bf),len(Q)); latn=rng.permutation(Q.lat.values)
Xp=build(tq,Q.lat.values,S,S.ay.values); Xn=build(tb.iloc[idx],latn,Bf.iloc[idx].reset_index(drop=True),Bf.ay.values[idx])
y=np.r_[np.ones(len(Q)),np.zeros(len(Q))]; models={}
for k in "ABC":
    rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=5,random_state=0,n_jobs=-1); rf.fit(np.r_[Xp[k],Xn[k]],y); models[k]=rf
print("trained on",len(Q),"+",len(Q))
# ---------------- test sets
def make(name,Qh,y0,y1):
    t=pd.to_datetime(Qh.datetime).dt.tz_localize(None); sid=M.enrich(Qh); ay=M.ayan(jd(t.values)); lat=Qh.latitude.values if "latitude" in Qh else Qh.lat.values
    G=grid(y0,y1); gs=M.enrich(G); gay=M.ayan(jd(G.datetime.values)); return dict(name=name,t=t,lat=lat,sid=sid,ay=ay,G=G,gs=gs,gay=gay)
def evaluate(T,ndraw=10):
    Xe=build(T["t"],T["lat"],T["sid"],T["ay"]); out=[]
    for k in "ABC":
        aucs=[]; rg=np.random.default_rng(21)
        for d in range(ndraw):
            ii=rg.integers(0,len(T["G"]),len(T["t"])); ln=rg.permutation(T["lat"])
            Xg=build(T["G"].datetime.iloc[ii],ln,T["gs"].iloc[ii].reset_index(drop=True),T["gay"][ii])
            Xt=np.r_[Xe[k],Xg[k]]; yt=np.r_[np.ones(len(T["t"])),np.zeros(len(T["t"]))]; aucs.append(roc_auc_score(yt,models[k].predict_proba(Xt)[:,1]))
        # bootstrap over events using the last draw's scores
        s=models[k].predict_proba(Xt)[:,1]; n=len(T["t"]); sp,sn=s[:n],s[n:]; bs=[]
        for _ in range(1000):
            a=sp[rg.integers(0,n,n)]; b=sn[rg.integers(0,n,n)]; bs.append(stats_auc(a,b))
        # permutation test
        nulls=[]
        for _ in range(200):
            pm=rg.permutation(len(yt)); nulls.append(roc_auc_score(yt[pm],s))
        out.append((T["name"],k,n,round(float(np.mean(aucs)),4),round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4),float((np.array(nulls)>=np.mean(aucs)).mean())))
    return out
def stats_auc(a,b):
    allv=np.r_[a,b]; rk=pd.Series(allv).rank().values; return (rk[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))
res=[]
for nm,f,y0,y1 in (("Era A 1964-1989 M5.5+","histA_declustered.csv",1964,1989),("Era B 1900-1963 M7.0+","histB_declustered.csv",1900,1963)):
    Qh=pd.read_csv(f); res+=evaluate(make(nm,Qh,y0,y1))
rp=pd.read_csv("../../1850-1950-EQData-MAG5.csv"); rp=rp[(rp.type=="earthquake")&(rp.mag>=5.5)].copy()
rp=rp[pd.to_datetime(rp.time).dt.tz_localize(None).dt.year.between(1850,1950)]; print("repo raw M5.5+",len(rp)); Dr=decluster(rp); print("repo declustered",len(Dr))
Dr[["id","datetime","latitude","longitude","mag","depth"]].to_csv("repo_1850_1950_declustered.csv",index=False)
res+=evaluate(make("Repo file 1850-1950 M5.5+",Dr,1850,1950))
R=pd.DataFrame(res,columns=["test_set","model","n_events","mean_AUC","AUC_CI_low","AUC_CI_high","perm_p"]); R["passes_primary"]=((R.model=="A")&(R.test_set.str.startswith("Era A"))&(R.mean_AUC>=0.52)&(R.AUC_CI_low>0.5))
print(R.to_string()); R.to_csv("tgsai_historical_results.csv",index=False)
