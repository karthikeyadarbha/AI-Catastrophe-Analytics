import sys,numpy as np,pandas as pd
sys.path.insert(0,"../cat"); import myeph as M
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
r=np.radians
d=pd.read_csv("train.csv"); d['t']=pd.to_datetime(d.time,format='ISO8601',utc=True).dt.tz_localize(None)
d['jd']=2440587.5+(d.t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
e=d[d.t.dt.year>=1850].reset_index(drop=True); n=len(e)
MD=[(0,0,1,0,-20905355),(2,0,-1,0,-3699111),(2,0,0,0,-2955968),(0,0,2,0,-569925),(0,1,0,0,48888),(0,0,0,2,-3149),(2,0,-2,0,246158),(2,-1,-1,0,-152138),(2,0,1,0,-170733),(2,-1,0,0,-204586),(0,1,-1,0,-129620),(1,0,0,0,108743),(0,1,1,0,104755)]
def mdist(J):
    T=(J-2451545.0)/36525; D=r(297.8501921+445267.1114034*T); Ms=r(357.5291092+35999.0502909*T); Mp=r(134.9633964+477198.8675055*T); F=r(93.2720950+483202.0175233*T)
    return 385000.56+sum(c*np.cos(a*D+b*Ms+cc*Mp+f*F) for a,b,cc,f,c in MD)/1000
def raw(J,lat):
    A=M.all_trop(J); ay=M.ayan(J); sid={p:(A[p]-ay)%360 for p in ["Sun","Moon","Mars","Jupiter","Saturn","Mercury","Venus"]}
    node_t=M.node_trop(J); node_s=(node_t-ay)%360
    mt=A["Moon"]; dec=np.degrees(np.arcsin(np.sin(r(23.439))*np.sin(r(mt)))) # ecliptic-latitude-free approx
    el=(sid["Moon"]-sid["Sun"])%360; ph=np.minimum(el,360-el)
    return sid,node_s,node_t,dec,ph,mdist(J)
def vedic(sid):
    nk=(sid["Moon"]//(360/27)).astype(int); oh=np.eye(27)[nk]
    el=(sid["Moon"]-sid["Sun"])%360
    return np.c_[oh[:,[14,22,0,11,12,13]].sum(1),oh[:,[2,15,24,1,19,9,7,10]].sum(1),oh[:,[17,20,21,3,4,6,16]].sum(1),oh[:,[23,25,26,5,8,18]].sum(1),oh[:,[5,8,17,18]].sum(1),np.floor(el/12)]
sc=lambda x:[np.sin(r(x)),np.cos(r(x))]
def X(J,lat,node="sid",vd=False):
    sid,ns,nt,dec,ph,dist=raw(J,lat)
    cols=[np.abs(dec),np.abs(dec)/10,ph,dist]
    for p in ["Moon","Sun","Mars","Jupiter","Saturn"]: cols+=sc(sid[p])
    cols+=sc(ns if node=="sid" else nt)
    M_=np.c_[tuple(cols)]
    return np.c_[M_,vedic(sid)] if vd else M_
def run(label,node="sid",vd=False,reps=10):
    Xp=X(e.jd.values,e.latitude.values,node,vd); out=[];tsp=[];sh=[]
    yr=e.t.dt.year.values
    for rep in range(reps):
        g=np.random.default_rng(rep); Jn=g.uniform(e.jd.min(),e.jd.max(),n*20); Xn=X(Jn,None,node,vd)
        Xa=np.r_[Xp,Xn]; ya=np.r_[np.ones(n),np.zeros(len(Xn))]
        cv=StratifiedKFold(5,shuffle=True,random_state=rep); mk=lambda:RandomForestClassifier(300,min_samples_leaf=3,class_weight="balanced",random_state=rep,n_jobs=-1)
        out.append(roc_auc_score(ya,cross_val_predict(mk(),Xa,ya,cv=cv,method="predict_proba")[:,1]))
        yrn=pd.to_datetime((Jn-2440587.5)*86400,unit="s").year.values; ay_=np.r_[yr,yrn]; tr=ay_<1970; te=~tr
        m=mk().fit(Xa[tr],ya[tr]); tsp.append(roc_auc_score(ya[te],m.predict_proba(Xa[te])[:,1]))
        ys=g.permutation(ya); sh.append(roc_auc_score(ys,cross_val_predict(mk(),Xa,ys,cv=cv,method="predict_proba")[:,1]))
    print(f"{label:48s} CV {np.mean(out):.3f} ({min(out):.3f}-{max(out):.3f})  time-split {np.mean(tsp):.3f}  shuffled {np.mean(sh):.3f}/{max(sh):.3f}")
    return np.mean(out),min(out),max(out),np.mean(tsp),np.mean(sh),max(sh)
print("events",n)
res={}
res["fixed"]=run("Fixed: true planets, sidereal node, true distance")
res["tropnode"]=run("Same but node left tropical (original frame)",node="trop")
res["vedic"]=run("Fixed + Vedic features (mandalas, sharp, tithi)",vd=True)
pd.DataFrame(res,index=["cv_mean","cv_min","cv_max","time_split","shuf_mean","shuf_max"]).T.round(3).to_csv("refit.csv"); print(open("refit.csv").read())
