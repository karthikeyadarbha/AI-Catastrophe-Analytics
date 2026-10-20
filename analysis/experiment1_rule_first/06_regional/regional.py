import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from scipy import stats
from crescent import *
from hist import grid
W=360/27; nk=lambda x:(x//W).astype(int)
REG=[("Japan-Kurils-Korea",30,52,128,160),("Philippines-Taiwan",6,26,116,128),("Indonesia",-12,6,94,141),("China",20,42,100,128),("Himalaya-India-Myanmar",5,36,75,100),("West-Central Asia",24,42,26,75)]
def assign(lat,lon):
    out=np.array(["Rest"]*len(lat),dtype=object)
    for n,a,b,c,d in REG[::-1]: out[(lat>=a)&(lat<=b)&(lon>=c)&(lon<=d)]=n
    return out
def mk(d):
    o={}
    for n,l in d.items():
        for k in l: o[k]=n
    return o
BS=mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]}); PT=mk({"Anila":[14,22,0,11,12,13],"Anala":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]})
SHARP=[5,8,17,18]
def sky(S):
    L=lambda p:S[p+"_Lon"].values; oth=np.vstack([L(p) for p in ["Sun","Mars","Mercury","Jupiter","Venus","Saturn"]]).T
    n=nk(L("Moon")); return dict(bs=np.array([BS[i] for i in n]),pt=np.array([PT[i] for i in n]),r10=(np.abs((L("Moon")[:,None]-oth+180)%360-180)<=5).any(1),sharp=np.isin(n,SHARP),vish=(L("Sun")>=200)&(L("Sun")<200+W))
# Ujjain-style monthly flags for each region centre
k=np.arange(int((1899-2000)*12.3685),int((2029-2000)*12.3685)); tcm=np.sort(conj_before(2451550.09766+29.530588861*k+1.0))
CEN={n:((a+b)/2,(c+d)/2) for n,a,b,c,d in REG}; CF={}
for n,(la,lo) in CEN.items():
    f,t,a=sight(tcm,np.full(len(tcm),la),np.full(len(tcm),lo)); CF[n]=flags(f,t,a)[0]
lookup=lambda jd:np.searchsorted(tcm,jd,side="right")-1
def chunked(tc,lat,lon,n=4000):
    F=[];T=[];A=[]
    for i in range(0,len(tc),n):
        f,t,a=sight(tc[i:i+n],lat[i:i+n],lon[i:i+n]); F.append(f);T.append(t);A.append(a)
    return flags(np.concatenate(F),np.concatenate(T),np.concatenate(A))[0]
rng=np.random.default_rng(29)
def evaluate(label,t,lat,lon,Sev,Sbg,gj):
    reg=assign(lat,lon); jd=jd_of(t); SK=sky(Sev); SB=sky(Sbg); rows=[]
    yoke_e=chunked(conj_before(jd),lat,lon)
    # crescent background: 3 random grid times per event at the event's own location
    yb=[]
    for _ in range(3): ii=rng.integers(0,len(gj),len(jd)); yb.append(chunked(conj_before(gj[ii]),lat,lon))
    for name in [r[0] for r in REG]+["Rest"]:
        m=reg==name; n=int(m.sum())
        if n<10: continue
        def chi(rn,e,b):
            lv=sorted(set(b)); sh=np.array([(b==l).mean() for l in lv]); o=np.array([(e[m]==l).sum() for l in lv]); rows.append((label,name,rn,n,"/".join(map(str,o)),"/".join(f"{x:.0f}" for x in sh*n),round(float(np.max(o/(sh*n))),3),stats.chisquare(o,sh*n)[1]))
        def bino(rn,e,p0): o=int(e[m].sum()); ex=n*p0; rows.append((label,name,rn,n,str(o),f"{ex:.1f}",round(o/ex,3),stats.binom.sf(o-1,n,p0)))
        chi("R1 BS circle",SK["bs"],SB["bs"]); chi("R2 PT mandala",SK["pt"],SB["pt"]); bino("R3 Moon near Sun/planet",SK["r10"],SB["r10"].mean()); bino("R4 Moon in sharp group",SK["sharp"],SB["sharp"].mean()); bino("R5 Sun in Vishakha",SK["vish"],SB["vish"].mean())
        bino("R6 level yoke, epicentre observer",yoke_e,np.concatenate([y[m] for y in yb]).mean())
        if name!="Rest":
            flag=CF[name]; obs=flag[lookup(jd[m])].mean(); nul=np.array([flag[lookup(np.clip(jd[m]+int(rng.integers(-3650,3651)),tcm[0],tcm[-1]))].mean() for _ in range(500)])
            rows.append((label,name,"R7 level yoke, region-centre observer",n,round(obs,4),round(nul.mean(),4),round(obs/nul.mean(),3),(np.sum(nul>=obs)+1)/501))
    return rows
allr=[]
Q=pd.read_csv("quakes_m55_input.csv"); S=pd.read_csv("m55_features_sid.csv"); Bf=pd.read_csv("bg_features_myeph.csv"); bgd=pd.read_csv("bg_sid.csv")
tq=pd.to_datetime(Q.datetime).dt.tz_localize(None); tb=pd.to_datetime(bgd.datetime).dt.tz_localize(None).reset_index(drop=True); sp=(tq>=pd.Timestamp("2008-01-01")).values; bsp=(tb>=pd.Timestamp("2008-01-01")).values
for lab,m,bm in (("1990-2007 discovery",~sp,~bsp),("2008-2026 holdout",sp,bsp)):
    allr+=evaluate(lab,tq[m].reset_index(drop=True),Q.lat.values[m],Q.lon.values[m],S[m].reset_index(drop=True),Bf[bm].reset_index(drop=True),jd_of(tb[bm])); print(lab,"done",flush=True)
import myeph as M
for lab,fn,y0,y1 in (("Era A 1964-1989","histA_declustered.csv",1964,1989),("Era B 1900-1963 (M7+)","histB_declustered.csv",1900,1963)):
    h=pd.read_csv(fn); th=pd.to_datetime(h.datetime).dt.tz_localize(None); G=grid(y0,y1)
    allr+=evaluate(lab,th,h.latitude.values,h.longitude.values,M.enrich(h),M.enrich(G),jd_of(G.datetime)); print(lab,"done",flush=True)
R=pd.DataFrame(allr,columns=["data_set","region","rule","n","observed","expected","ratio","p"]); R["p"]=R.p.astype(float); R["bar"]=0.05/42
R.to_csv("regional_results.csv",index=False)
pd.set_option("display.width",230); pd.set_option("display.max_colwidth",36)
fam=R[R.region!="Rest"]; h=fam[fam.data_set=="2008-2026 holdout"]; print("holdout tests",len(h),"min p",h.p.min().round(5),"count p<0.05",(h.p<0.05).sum(),"expected by chance",round(len(h)*0.05,1))
print(h.sort_values("p").head(8)[["region","rule","n","observed","expected","ratio","p"]].round(4).to_string())
for d in ["1990-2007 discovery","Era A 1964-1989","Era B 1900-1963 (M7+)"]:
    x=fam[fam.data_set==d]; print(d,"tests",len(x),"min p",x.p.min().round(5),"p<0.05:",(x.p<0.05).sum(),"p<bar:",(x.p<0.05/42).sum())
