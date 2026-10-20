import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
sys.path.insert(0,"../shared"); import myeph as M
from hist import grid
P7=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]; W=360/27; nk=lambda x:(x//W).astype(int)
def cats(df):
    L={p:df[f"{p}_Lon"].values for p in P7}; C={}
    for p in P7: C[f"{p} nakshatra"]=(nk(L[p]),27)
    for p in P7: C[f"{p} sign"]=((L[p]//30).astype(int),12)
    C["Moon tithi"]=((((L["Moon"]-L["Sun"])%360)//12).astype(int),30)
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]:
        C[f"Moon-{p} separation"]=((((L["Moon"]-L[p])%360)//30).astype(int),12); C[f"Sun-{p} separation"]=((((L["Sun"]-L[p])%360)//30).astype(int),12)
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: C[f"{p} retro"]=(df[p+"_Is_Retro"].astype(int).values,2)
    return C
keys=None
def zvec(CQ,CB):
    out=[]
    for k,(a,m) in CQ.items():
        pb=np.bincount(CB[k][0],minlength=m)/len(CB[k][0]); o=np.bincount(a,minlength=m); e=pb*len(a); out.append((o-e)/np.sqrt(np.maximum(e*(1-pb),1e-9)))
    return out
S=pd.read_csv("m55_features_sid.csv"); B=pd.read_csv("bg_features_myeph.csv"); bgs=pd.read_csv("bg_sid.csv")
Q=pd.read_csv("quakes_m55_input.csv")
CQ0,CB0=cats(S),cats(B); z0=np.concatenate(zvec(CQ0,CB0)); keys=list(CQ0)
QA=pd.read_csv("histA_declustered.csv"); SA=M.enrich(QA); G=grid(1964,1989); BA=M.enrich(G)
CQA,CBA=cats(SA),cats(BA); zA=np.concatenate(zvec(CQA,CBA)); r=np.corrcoef(z0,zA)[0,1]
rng=np.random.default_rng(5); nul=[]; nA=len(QA)
for _ in range(300):
    ii=rng.integers(0,len(BA),nA); CQn={k:(v[0][ii],v[1]) for k,v in CBA.items()}; nul.append(np.corrcoef(z0,np.concatenate(zvec(CQn,CBA)))[0,1])
nul=np.array(nul); p=(nul>=r).mean(); print("bins",len(z0),"r",round(r,4),"null mean",nul.mean().round(4),"sd",nul.std().round(4),"p",p)
lab=[(k,i) for k in keys for i in range(CQ0[k][1])]
D=pd.DataFrame({"feature":[a for a,_ in lab],"bin":[b for _,b in lab],"z_1990_2026":z0,"z_1964_1989":zA})
top=D.reindex(D.z_1990_2026.abs().sort_values(ascending=False).index).head(8); print(top.round(2).to_string())
rep=D[(D.z_1990_2026.abs()>=2.5)&(D.z_1964_1989.abs()>=2.5)&(np.sign(D.z_1990_2026)==np.sign(D.z_1964_1989))]; print("bins with |z|>=2.5 in both and same sign:",len(rep)); print(rep.round(2).to_string())
print("expected by chance among",(D.z_1990_2026.abs()>=2.5).sum(),"strong 1990-2026 bins: same-sign |z|>=2.5 prob ~",round(0.0062/2,4))
D.to_csv("hist_bin_scan_comparison.csv",index=False); pd.DataFrame([dict(r=r,null_mean=nul.mean(),null_sd=nul.std(),p=p,n_bins=len(z0),n_events=nA)]).to_csv("hist_bin_scan_summary.csv",index=False)
