exec(open("refit.py").read().split('print("events",n)')[0])
q=pd.read_csv(O if False else "/mnt/user-data/outputs/earthquakes_parashara_tantra_mandalas.csv"); print(q.columns.tolist()[:8])
tc=[c for c in q.columns if c.lower() in("utc",)][0]; lc=[c for c in q.columns if c.lower() in("lat","latitude")][0]
q['t']=pd.to_datetime(q[tc],format='ISO8601',utc=True).dt.tz_localize(None); q['jd']=2440587.5+(q.t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
ov=np.array([np.any(abs(e.jd.values-j)<2) for j in q.jd]); print("overlap with training file:",q.loc[ov,'Event'].tolist())
tr=e[~np.array([np.any(abs(q.jd.values-j)<2) for j in e.jd])].reset_index(drop=True)
Xp=X(tr.jd.values,None); Xq=X(q.jd.values,None)
aucs=[];pcts=[]
for rep in range(10):
    g=np.random.default_rng(rep); Jn=g.uniform(tr.jd.min(),tr.jd.max(),len(tr)*20); Xn=X(Jn,None)
    m=RandomForestClassifier(300,min_samples_leaf=3,class_weight="balanced",random_state=rep,n_jobs=-1).fit(np.r_[Xp,Xn],np.r_[np.ones(len(Xp)),np.zeros(len(Xn))])
    Jt=g.uniform(2451545-9*365.25,2461000,3000); st=m.predict_proba(X(Jt,None))[:,1]; sq=m.predict_proba(Xq)[:,1]
    aucs.append(roc_auc_score(np.r_[np.ones(len(sq)),np.zeros(len(st))],np.r_[sq,st])); pcts.append([(st<s).mean() for s in sq])
print("trained on",len(tr),"great quakes (15-event overlaps removed); AUC of the 15 vs random times: mean %.3f range %.3f-%.3f"%(np.mean(aucs),min(aucs),max(aucs)))
pc=np.mean(pcts,axis=0); q['pct']=pc; print(q.assign(pct=(pc*100).round(0))[['Event','pct']].to_string()); print("share above 50th pct:",(pc>.5).sum(),"of 15; above 75th:",(pc>.75).sum())
pd.DataFrame({"event":q['Event'],"score_percentile_vs_random_times":(pc*100).round(0)}).to_csv("score15.csv",index=False)
