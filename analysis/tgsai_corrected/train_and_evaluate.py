"""Train and evaluate corrected TG-SAI models against matched random negatives.
usage: python train_and_evaluate.py TRAINING.csv [--score EVENTS.csv] [--reps 10]
 - positives: earthquakes in TRAINING.csv (1850+); negatives: random times in the same years (20 per positive)
 - reports 5-fold CV AUC, a time split (train before 1970), and shuffled-label chance level
 - optional --score EVENTS.csv (columns: Event, UTC): re-score listed events after removing training rows within 2 days of them"""
import argparse,numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import features as F
ap=argparse.ArgumentParser(); ap.add_argument("training"); ap.add_argument("--score"); ap.add_argument("--reps",type=int,default=10); a=ap.parse_args()
d=pd.read_csv(a.training); d["jd"]=F.to_jd(d["time"]); d=d[d.jd>=2396758.5].reset_index(drop=True); n=len(d)
yr=pd.to_datetime((d.jd.values-2440587.5)*86400,unit="s").year.values
mk=lambda s:RandomForestClassifier(300,min_samples_leaf=3,class_weight="balanced",random_state=s,n_jobs=-1)
def run(label,**kw):
    Xp=F.feature_matrix(d.jd.values,**kw); cv_,ts_,sh_=[],[],[]
    for rep in range(a.reps):
        g=np.random.default_rng(rep); Jn=g.uniform(d.jd.min(),d.jd.max(),n*20); Xn=F.feature_matrix(Jn,**kw)
        X=np.r_[Xp,Xn]; y=np.r_[np.ones(n),np.zeros(len(Xn))]; cv=StratifiedKFold(5,shuffle=True,random_state=rep)
        cv_.append(roc_auc_score(y,cross_val_predict(mk(rep),X,y,cv=cv,method="predict_proba")[:,1]))
        yrs=np.r_[yr,pd.to_datetime((Jn-2440587.5)*86400,unit="s").year.values]; tr=yrs<1970
        ts_.append(roc_auc_score(y[~tr],mk(rep).fit(X[tr],y[tr]).predict_proba(X[~tr])[:,1]))
        ys=g.permutation(y); sh_.append(roc_auc_score(ys,cross_val_predict(mk(rep),X,ys,cv=cv,method="predict_proba")[:,1]))
    print(f"{label:52s} CV {np.mean(cv_):.3f} ({min(cv_):.3f}-{max(cv_):.3f}) time-split {np.mean(ts_):.3f} shuffled {np.mean(sh_):.3f}/{max(sh_):.3f}")
print("positives:",n)
run("Fixed features"); run("Node left tropical (original frame)",node_frame="tropical"); run("Fixed + Vedic features",vedic=True)
if a.score:
    q=pd.read_csv(a.score); q["jd"]=F.to_jd(q["UTC"]); keep=~np.array([np.any(abs(q.jd.values-j)<2) for j in d.jd]); tr=d[keep]
    Xp=F.feature_matrix(tr.jd.values); Xq=F.feature_matrix(q.jd.values); aucs=[]; pct=[]
    for rep in range(a.reps):
        g=np.random.default_rng(rep); Jn=g.uniform(tr.jd.min(),tr.jd.max(),len(tr)*20)
        m=mk(rep).fit(np.r_[Xp,F.feature_matrix(Jn)],np.r_[np.ones(len(Xp)),np.zeros(len(Jn))])
        Jt=g.uniform(2451545-9*365.25,2461000,3000); st=m.predict_proba(F.feature_matrix(Jt))[:,1]; sq=m.predict_proba(Xq)[:,1]
        aucs.append(roc_auc_score(np.r_[np.ones(len(sq)),np.zeros(len(st))],np.r_[sq,st])); pct.append([(st<s).mean() for s in sq])
    print(f"Re-score of {len(q)} listed events: AUC {np.mean(aucs):.3f} ({min(aucs):.3f}-{max(aucs):.3f})")
    for e,p in zip(q["Event"],np.mean(pct,axis=0)): print(f"  {e:20s} percentile {100*p:.0f}")
