import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
Q=pd.read_csv("results/declustered_events_M65.csv"); Q["t"]=pd.to_datetime(Q.time).dt.tz_localize(None)
sid=pd.read_csv("quakes_m65_sid.csv").set_index("id"); trop=pd.read_csv("quakes_m65_trop.csv").set_index("id")
bt=pd.read_csv("bg_trop.csv"); bt["t"]=pd.to_datetime(bt.datetime).dt.tz_localize(None)
bt=bt[(bt.t>="1990-01-01")&(bt.t<="2026-10-16")].reset_index(drop=True)
rng=np.random.default_rng(5)
def jd(ts): return (ts.astype("datetime64[s]").astype(np.int64))/86400.0+2440587.5
def repo_feats(J,lat,planets="crude",P=None):
    T=(J-2451545.0)/36525.0; r=np.radians
    sun=(280.466+36000.77*T+1.915*np.sin(r(357.529+35999.05*T)))%360
    sdec=np.degrees(np.arcsin(np.sin(r(23.439))*np.sin(r(sun))))
    Mm=(134.963+477198.8676*T)%360
    moon=(218.316+481267.88*T+6.289*np.sin(r(Mm)))%360
    mlat=5.128*np.sin(r(93.27+483202.01*T))
    mdec=np.degrees(np.arcsin(np.sin(r(mlat))*np.cos(r(23.439))+np.cos(r(mlat))*np.sin(r(23.439))*np.sin(r(moon))))
    if planets=="crude":
        pl=lambda Lm,pi,e:(Lm+np.degrees(2*e*np.sin(r(Lm-pi))))%360
        sat=pl(50.08+1222.1*T,92.06,0.0555); jup=pl(34.35+3034.9*T,14.33,0.0485); mars=pl(355.43+19140.3*T,336.06,0.0934)
        sunL=sun
    else:  # true geocentric from the ephemeris tool (tropical frame), Sun from the same source
        sat,jup,mars,sunL=P["Saturn"],P["Jupiter"],P["Mars"],P["Sun"]; moon=P["Moon"]
    al=lambda a,b:(1+np.cos(r(2*(a-b))))/2
    return np.c_[al(moon,sunL),al(sat,sunL),al(jup,sunL),al(mars,sunL),(1+np.cos(r(Mm)))/2,np.cos(r(lat-mdec)),np.cos(r(lat-sdec))]
def PT(P_sid,P_trop):
    W=360/27; nk=(P_sid["Moon"]//W).astype(int)
    el=(P_sid["Moon"]-P_sid["Sun"])%360
    oh=np.eye(27)[nk]
    return np.c_[np.sin(np.radians(P_sid["Moon"])),np.cos(np.radians(P_sid["Moon"])),np.sin(np.radians(el)),np.cos(np.radians(el)),oh[:,[14,22,0,11,12,13]].sum(1),oh[:,[2,15,24,1,19,9,7,10]].sum(1),oh[:,[17,20,21,3,4,6,16]].sum(1),oh[:,[23,25,26,5,8,18]].sum(1)]
cols=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
# positives
pos_t=Q.t.values; pos_lat=Q.latitude.values
Pp_trop={c:trop.loc[Q.id,f"{c}_Lon"].values for c in cols}; Pp_sid={c:sid.loc[Q.id,f"{c}_Lon"].values for c in cols}
# negatives: random background rows (random time), latitude copied from a randomly chosen event (matched latitude distribution)
idx=rng.integers(0,len(bt),len(Q)); lat_n=rng.permutation(pos_lat)
neg_t=bt.t.values[idx]
Pn_trop={c:bt.loc[idx,f"{c}_Lon"].values for c in cols}
bs=pd.read_csv("bg_sid.csv"); bs=bs[(pd.to_datetime(bs.datetime).dt.tz_localize(None)>="1990-01-01")&(pd.to_datetime(bs.datetime).dt.tz_localize(None)<="2026-10-16")].reset_index(drop=True)
Pn_sid={c:bs.loc[idx,f"{c}_Lon"].values for c in cols}
y=np.r_[np.ones(len(Q)),np.zeros(len(Q))]; tall=np.r_[pos_t,neg_t]; late=np.r_[pos_lat,lat_n]
J=jd(tall)
A=repo_feats(J,late,"crude")
Ptr={c:np.r_[Pp_trop[c],Pn_trop[c]] for c in cols}; Psi={c:np.r_[Pp_sid[c],Pn_sid[c]] for c in cols}
B=repo_feats(J,late,"true",Ptr)
C=np.c_[B,PT(Psi,Ptr)]
yrs=pd.to_datetime(tall).year.values
def evaluate(X,name):
    rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=5,random_state=0,n_jobs=-1)
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    p=cross_val_predict(rf,X,y,cv=cv,method="predict_proba")[:,1]; a1=roc_auc_score(y,p)
    tr=yrs<2008; rf.fit(X[tr],y[tr]); a2=roc_auc_score(y[~tr],rf.predict_proba(X[~tr])[:,1])
    # permutation null for the CV AUC
    nulls=[]
    for s in range(30):
        yp=np.random.default_rng(100+s).permutation(y)
        pp=cross_val_predict(rf,X,yp,cv=cv,method="predict_proba")[:,1]; nulls.append(roc_auc_score(yp,pp))
    return (name,X.shape[1],round(a1,3),round(a2,3),round(np.mean(nulls),3),round(np.max(nulls),3))
R=pd.DataFrame([evaluate(A,"A: repo formulas (crude planets), instantaneous"),evaluate(B,"B: true geocentric planets, instantaneous"),evaluate(C,"C: B + Vedic features (Moon, tithi, mandala)")],
 columns=["model","n_features","AUC_5fold_CV","AUC_train<2008_test>=2008","AUC_shuffled_label_mean","AUC_shuffled_label_max"])
print(R.to_string()); R.to_csv("results/tgsai_retrain_results.csv",index=False)
print("n pos",len(Q),"n neg",len(Q))
