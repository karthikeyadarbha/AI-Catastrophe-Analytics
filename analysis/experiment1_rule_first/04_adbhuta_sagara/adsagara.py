import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import myeph as M
from hist import grid
W=360/27; nk=lambda x:(x//W).astype(int); r=np.radians
def mk(d):
    o={}
    for name,l in d.items():
        for k in l: o[k]=name
    return o
LV=["Vayu","Agni","Indra","Varuna"]
PTQ=mk({"Vayu":[14,22,0,11,12,13],"Agni":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]})
BRH=mk({"Vayu":[11,12,13,14,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[17,16,21,22,6,3,20],"Varuna":[26,19,5,8,18,25,23]})
CMP=mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[10,7,2,15,1,9],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]})
BSL=mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]})
PTX=mk({"Vayu":[14,22,0,11,12,13],"Agni":[2,15,24,1,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]})
DISP=[4,6,19,22]; BH={"Vayu":120,"Agni":90,"Indra":80,"Varuna":70}
def solar(t,lat,lon):
    jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
    lam=M.sun_trop(jd); dec=np.arcsin(np.sin(r(23.439))*np.sin(r(lam)))
    h=(t.dt.hour.values+t.dt.minute.values/60+t.dt.second.values/3600+lon/15)%24
    x=-np.tan(r(lat))*np.tan(dec); ok=np.abs(x)<1; H0=np.degrees(np.arccos(np.clip(x,-1,1)))/15
    return h,12-H0,12+H0,ok
def yama(h,sr,ss,ok,same=False):
    out=np.array(["NA"]*len(h),dtype=object); day=(ss-sr); night=24-day
    dd=(h>=sr)&(h<ss)&ok; k=np.floor((h-sr)/(day/4)).astype(int).clip(0,3)
    out[dd]=np.array(LV)[k[dd]]
    nn=(~dd)&ok; tn=(h-ss)%24; k2=np.floor(tn/(night/4)).astype(int).clip(0,3); out[nn]=np.array(LV if same else LV[::-1])[k2[nn]]
    return out
def run(label,t,lat,lon,mag,moonE,tb,moonB):
    rng=np.random.default_rng(11); rows=[]; n=len(t)
    tb=(tb+pd.to_timedelta(rng.uniform(0,6*3600,len(tb)),unit='s')).reset_index(drop=True); moonB=M.enrich(pd.DataFrame({'datetime':tb}))['Moon_Lon'].values
    ix=rng.integers(0,n,len(tb)); latb,lonb=lat[ix],lon[ix]
    he,sre,sse,oke=solar(t,lat,lon); hb,srb,ssb,okb=solar(tb,latb,lonb)
    YE,YB=yama(he,sre,sse,oke),yama(hb,srb,ssb,okb); YE2,YB2=yama(he,sre,sse,oke,True),yama(hb,srb,ssb,okb,True)
    nE,nB=nk(moonE),nk(moonB)
    cE={k:np.array([m.get(i,"NA") for i in nE]) for k,m in (("PTX",PTX),("PTQ",PTQ),("BRH",BRH),("CMP",CMP),("BSL",BSL))}
    cB={k:np.array([m.get(i,"NA") for i in nB]) for k,m in (("PTX",PTX),("PTQ",PTQ),("BRH",BRH),("CMP",CMP),("BSL",BSL))}
    def chi(name,ce,cb):
        oe=np.isin(ce,LV); ob=np.isin(cb,LV); sh=np.array([(cb[ob]==l).mean() for l in LV]); o=np.array([(ce[oe]==l).sum() for l in LV]); nn=o.sum()
        rows.append((label,name,int(nn),"/".join(map(str,o)),"/".join(f"{x:.0f}" for x in sh*nn),round(float(np.max(o/(sh*nn))),3),stats.chisquare(o,sh*nn)[1]))
    def bino(name,e_,b_,ve=None,vb=None):
        e_=np.asarray(e_); b_=np.asarray(b_)
        if ve is not None: e_=e_[ve]; b_=b_[vb]
        o=int(e_.sum()); p0=b_.mean(); ex=len(e_)*p0; rows.append((label,name,len(e_),str(o),f"{ex:.1f}",round(o/ex,3),stats.binom.sf(o-1,len(e_),p0)))
    chi("T1 circles, Brihaspati lists",cE["BRH"],cB["BRH"]); chi("T2 circles, compiler's summary lists",cE["CMP"],cB["CMP"]); chi("T3 circles, Parashara as quoted",cE["PTQ"],cB["PTQ"])
    bino("T4 Moon in a disputed nakshatra",np.isin(nE,DISP),np.isin(nB,DISP))
    chi("T5 yama-clock circle (day forward, night reversed)",YE,YB)
    vE=np.isin(YE,LV); vB=np.isin(YB,LV)
    bino("T6 yama circle = nakshatra circle (BS lists)",YE==cE["BSL"],YB==cB["BSL"],vE,vB)
    bino("T7 yama circle = nakshatra circle (Brihaspati lists)",YE==cE["BRH"],YB==cB["BRH"],vE,vB)
    tw=lambda h,sr,ss,ok:(np.minimum(abs(h-sr),abs(h-ss))<=0.8)&ok
    bino("T8 twilight (within 48 min of sunrise or sunset)",tw(he,sre,sse,oke),tw(hb,srb,ssb,okb),oke,okb)
    chi("T3b circles, Parashara exactly as quoted (P.Ashadha unlisted)",cE["PTX"],cB["PTX"])
    chi("T5b yama-clock circle (night in same order as day: Matsya, Vishnudharmottara)",YE2,YB2)
    vE2=np.isin(YE2,LV); vB2=np.isin(YB2,LV)
    bino("T6b same-order yama circle = nakshatra circle (BS lists)",YE2==cE["BSL"],YB2==cB["BSL"],vE2,vB2)
    bino("T7b same-order yama circle = nakshatra circle (Brihaspati lists)",YE2==cE["BRH"],YB2==cB["BRH"],vE2,vB2)
    ok=cE["BSL"]!="NA"; sc=np.array([BH[c] for c in cE["BSL"][ok]]); rho,p=stats.spearmanr(sc,mag[ok],alternative="greater"); rows.append((label,"T9 Bhargava felt-distance order vs magnitude",int(ok.sum()),f"rho={rho:.4f}","","",p))
    return rows
rows=[]
Q=pd.read_csv("quakes_m55_input.csv"); S=pd.read_csv("m55_features_sid.csv"); bg=pd.read_csv("bg_sid.csv"); B=pd.read_csv("bg_features_myeph.csv")
tq=pd.to_datetime(Q.datetime).dt.tz_localize(None); tb=pd.to_datetime(bg.datetime).dt.tz_localize(None).reset_index(drop=True)
sp=(tq>=pd.Timestamp("2008-01-01")).values; bsp=(tb>=pd.Timestamp("2008-01-01")).values
for lab,m,bm in (("Modern M5.5+ all","all","all"),("Modern M5.5+ 1990-2007 discovery",~sp,~bsp),("Modern M5.5+ 2008-2026 holdout",sp,bsp)):
    mm=np.ones(len(Q),bool) if isinstance(m,str) else m; bb=np.ones(len(B),bool) if isinstance(bm,str) else bm
    rows+=run(lab,tq[mm].reset_index(drop=True),Q.lat.values[mm],Q.lon.values[mm],Q.mag.values[mm],S.Moon_Lon.values[mm],tb[bb].reset_index(drop=True),B.Moon_Lon.values[bb])
for lab,f,y0,y1 in (("Older set 1 (1964-1989)","histA_declustered.csv",1964,1989),("Older set 2 (1900-1963)","histB_declustered.csv",1900,1963)):
    Qh=pd.read_csv(f); th=pd.to_datetime(Qh.datetime).dt.tz_localize(None); Sh=M.enrich(Qh); G=grid(y0,y1); Bh=M.enrich(G); tg=G.datetime.reset_index(drop=True)
    rows+=run(lab,th,Qh.latitude.values,Qh.longitude.values,Qh.mag.values,Sh.Moon_Lon.values,tg,Bh.Moon_Lon.values)
R=pd.DataFrame(rows,columns=["data_set","test","n","observed","expected","ratio","p"]); R["p"]=R.p.astype(float); R["threshold"]=round(0.05/13,4); R["passes"]=R.p<0.05/13
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",50); print(R.round(4).to_string()); R.to_csv("results/adbhuta_sagara_results.csv",index=False)
