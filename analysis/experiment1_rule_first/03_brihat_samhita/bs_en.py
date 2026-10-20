import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); sys.path.insert(0,"../shared"); import myeph as M; import eclipses as E
from hist import grid
W=360/27; nk=lambda x:(x//W).astype(int)
def mk(d):
    o={}
    for name,l in d.items():
        for k in l: o[k]=name
    return o
SAN=mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]})
E1=mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,23]})
E2=mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,16,21,22],"Varuna":[26,19,5,8,18,20,23]})
E3=mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,16,21,22],"Varuna":[26,19,5,8,18,23]})
LV=["Vayu","Agni","Indra","Varuna"]; DIST={"Vayu":200,"Varuna":180,"Indra":160,"Agni":110}
def tmand(h): return np.where(h<6,"Varuna",np.where(h<12,"Vayu",np.where(h<18,"Agni","Indra")))
def run(label,t,lon,mag,moonE,tb,moonB,eclE,eclB,split=None):
    rng=np.random.default_rng(3); rows=[]
    n=len(t); lonb=rng.permutation(np.resize(lon,len(tb)))   # shuffled event longitudes onto background times
    he=(t.dt.hour.values+t.dt.minute.values/60+lon/15)%24; hb=(tb.dt.hour.values+lonb/15)%24
    TE,TB=tmand(he),tmand(hb)
    def chi(name,ce,cb,levels):
        ok_e=np.isin(ce,levels); ok_b=np.isin(cb,levels); sh=np.array([(cb[ok_b]==l).mean() for l in levels]); o=np.array([(ce[ok_e]==l).sum() for l in levels]); nn=o.sum()
        rows.append((label,name,int(nn),"/".join(map(str,o)),"/".join(f"{x:.0f}" for x in sh*nn),round(float(np.max(o/(sh*nn))),3),stats.chisquare(o,sh*nn)[1]))
    def bino(name,ce,cb):
        o=int(ce.sum()); p0=cb.mean(); e=len(ce)*p0; rows.append((label,name,len(ce),str(o),f"{e:.1f}",round(o/e,3),stats.binom.sf(o-1,len(ce),p0)))
    nE,nB=nk(moonE),nk(moonB); mE={k:np.array([m.get(i,"NA") for i in nE]) for k,m in (("S",SAN),("E1",E1),("E2",E2),("E3",E3))}; mB={k:np.array([m.get(i,"NA") for i in nB]) for k,m in (("S",SAN),("E1",E1),("E2",E2),("E3",E3))}
    for k,nm in (("S","1 mandala, Sanskrit lists"),("E1","2 mandala, English printed (U.Ashadha in Indra)"),("E2","3 mandala, English printed (U.Ashadha in Varuna)"),("E3","4 mandala, both U.Ashadha and U.Bhadrapada set aside")): chi(nm,mE[k],mB[k],LV)
    chi("5 time-of-day circle (also v.27 precedence)",TE,TB,LV)
    bino("6 nakshatra circle = time circle",mE["S"]==TE,mB["S"]==TB)
    pe=np.array([a+"|"+b for a,b in zip(mE["S"],TE)]); pb=np.array([a+"|"+b for a,b in zip(mB["S"],TB)]); chi("7 all 16 circle pairs (v.28-31)",pe,pb,[a+"|"+b for a in LV for b in LV])
    for nm,cE in (("8 felt distance order by nakshatra circle",mE["S"]),("9 felt distance order by time circle",TE)):
        sc=np.array([DIST[c] for c in cE]); r,p=stats.spearmanr(sc,mag,alternative="greater"); rows.append((label,nm,n,f"rho={r:.4f}","","",p))
    bino("10 eclipse within 7 days before the earthquake",eclE,eclB); return rows
def ecl_before(t,ed,lo=0,hi=7):
    jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400; d=jd[:,None]-ed[None,:]; return ((d>lo)&(d<=hi)).any(1)
allrows=[]
# ---- 1990-2026
Q=pd.read_csv("quakes_m55_input.csv"); S=pd.read_csv("m55_features_sid.csv"); bg=pd.read_csv("bg_sid.csv"); B=pd.read_csv("bg_features_myeph.csv")
tq=pd.to_datetime(Q.datetime).dt.tz_localize(None); tb=pd.to_datetime(bg.datetime).dt.tz_localize(None).reset_index(drop=True)
ed=np.array([E.dt_to_jd(e["date"]) for e in E.find(1989,2028)])
eq=ecl_before(tq,ed); eb=ecl_before(tb,ed)
sp=(tq>=pd.Timestamp("2008-01-01")).values; bsp=(tb>=pd.Timestamp("2008-01-01")).values
for lab,m,bm in (("1990-2026 all",np.ones(len(Q),bool),np.ones(len(B),bool)),("1990-2007 discovery",~sp,~bsp),("2008-2026 holdout",sp,bsp)):
    allrows+=run(lab,tq[m].reset_index(drop=True),Q.lon.values[m],Q.mag.values[m],S.Moon_Lon.values[m],tb[bm].reset_index(drop=True),B.Moon_Lon.values[bm],eq[m],eb[bm])
# ---- historical eras
for lab,f,y0,y1 in (("Era A 1964-1989","histA_declustered.csv",1964,1989),("Era B 1900-1963 (M7+)","histB_declustered.csv",1900,1963)):
    Qh=pd.read_csv(f); th=pd.to_datetime(Qh.datetime).dt.tz_localize(None); Sh=M.enrich(Qh); G=grid(y0,y1); Bh=M.enrich(G); tg=G.datetime.reset_index(drop=True)
    edh=np.array([E.dt_to_jd(e["date"]) for e in E.find(y0-1,y1+1)])
    allrows+=run(lab,th,Qh.longitude.values,Qh.mag.values,Sh.Moon_Lon.values,tg,Bh.Moon_Lon.values,ecl_before(th,edh),ecl_before(tg,edh))
R=pd.DataFrame(allrows,columns=["data_set","test","n","observed","expected","ratio","p"]); R["p"]=R.p.astype(float); R["threshold"]=0.005; R["passes"]=(R.p<0.005)
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",58); print(R.round(4).to_string()); R.to_csv("bs_english_crosscheck_results.csv",index=False)
