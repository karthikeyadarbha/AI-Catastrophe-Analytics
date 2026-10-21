import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
"""Experiment 2 (Data-First): scan of every bin that Brihat Samhita (Sanskrit + English), Adbhuta Sagara and Parashara Tantra can define.
Discover on 1990-2007, check on 2008-2026, then look at the same bins in Older sets 1 and 2. Two-sided z, no direction pre-specified."""
import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import myeph as M; import eclipses as E
from hist import grid
W=360/27; nk=lambda x:(x//W).astype(int); rad=np.radians
def mk(d):
    o={}
    for name,l in d.items():
        for k in l: o[k]=name
    return o
LV=["Vayu","Agni","Indra","Varuna"]
SYS={
 "PT corrected":mk({"Vayu":[14,22,0,11,12,13],"Agni":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]}),
 "PT as quoted in AS":mk({"Vayu":[14,22,0,11,12,13],"Agni":[2,15,24,1,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]}),
 "BS Sanskrit":mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]}),
 "BS English v1":mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,23]}),
 "BS English v2":mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,16,21,22],"Varuna":[26,19,5,8,18,20,23]}),
 "BS English v3":mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,16,21,22],"Varuna":[26,19,5,8,18,23]}),
 "AS Brihaspati":mk({"Vayu":[11,12,13,14,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[17,16,21,22,6,3,20],"Varuna":[26,19,5,8,18,25,23]}),
 "AS compiler":mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[10,7,2,15,1,9],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]}),
}
DISP=[4,6,19,22]
def solar(t,lat,lon):
    jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
    lam=M.sun_trop(jd); dec=np.arcsin(np.sin(rad(23.439))*np.sin(rad(lam)))
    h=(t.dt.hour.values+t.dt.minute.values/60+t.dt.second.values/3600+lon/15)%24
    x=-np.tan(rad(lat))*np.tan(dec); ok=np.abs(x)<1; H0=np.degrees(np.arccos(np.clip(x,-1,1)))/15
    return h,12-H0,12+H0,ok
def yama(h,sr,ss,ok,same=False):
    out=np.array(["NA"]*len(h),dtype=object); day=(ss-sr); night=24-day
    dd=(h>=sr)&(h<ss)&ok; k=np.floor((h-sr)/(day/4)).astype(int).clip(0,3); out[dd]=np.array(LV)[k[dd]]
    nn=(~dd)&ok; tn=(h-ss)%24; k2=np.floor(tn/(night/4)).astype(int).clip(0,3); out[nn]=np.array(LV if same else LV[::-1])[k2[nn]]
    return out
def tmand(h): return np.where(h<6,"Varuna",np.where(h<12,"Vayu",np.where(h<18,"Agni","Indra")))
def ecl_before(t,ed):
    jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400; d=jd[:,None]-ed[None,:]; return ((d>0)&(d<=7)).any(1)
def lonlat(name,jd):
    T=(jd-2451545.0)/36525; X,Y,Z=M.helio(name,T); Xe,Ye,Ze=M.helio("Earth",T)
    dx,dy,dz=X-Xe,Y-Ye,Z-Ze; return np.degrees(np.arctan2(dy,dx))%360, np.degrees(np.arctan2(dz,np.hypot(dx,dy)))
def features(t,lat,lon,enr,ed):
    """returns dict name -> (labels array, levels, valid mask)"""
    F={}; n=len(t); nkm=nk(enr.Moon_Lon.values)
    cir={}
    for s,m in SYS.items():
        c=np.array([m.get(i,"NA") for i in nkm]); cir[s]=c; F[f"Moon circle, {s}"]=(c,LV,np.isin(c,LV))
    he,sr,ss,ok=solar(t,lat,lon)
    for nm,same in (("night reversed",False),("night same order",True)):
        y=yama(he,sr,ss,ok,same); F[f"Yama circle ({nm})"]=(y,LV,np.isin(y,LV))
        for s in ("BS Sanskrit","AS Brihaspati"):
            F[f"Yama circle = Moon circle, {s} ({nm})"]=(np.where(y==cir[s],"yes","no"),["yes","no"],np.isin(y,LV))
    F["Clock-time circle (BS Ch.32 6-hour blocks)"]=(tmand((he)),LV,np.ones(n,bool))
    F["Moon in a disputed nakshatra"]=(np.where(np.isin(nkm,DISP),"yes","no"),["yes","no"],np.ones(n,bool))
    F["Twilight (within 48 min of sunrise/sunset)"]=(np.where((np.minimum(abs(he-sr),abs(he-ss))<=0.8),"yes","no"),["yes","no"],ok)
    F["Eclipse in the 7 days before"]=(np.where(ecl_before(t,ed),"yes","no"),["yes","no"],np.ones(n,bool))
    jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
    lm,bm=lonlat("Mercury",jd); lj,bj=lonlat("Jupiter",jd); dl=((lm-lj+180)%360)-180; sep=np.hypot(dl*np.cos(rad((bm+bj)/2)),bm-bj)
    F["Mercury-Jupiter within 1 deg (BS 17 war)"]=(np.where(sep<1,"yes","no"),["yes","no"],np.ones(n,bool))
    F["Mercury-Jupiter within 1 deg, Mercury the southern one"]=(np.where((sep<1)&(bm<bj),"yes","no"),["yes","no"],np.ones(n,bool))
    pl=np.vstack([enr[p+"_Lon"].values for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]]).T; war=np.zeros(n,bool)
    for i in range(5):
        for j in range(i+1,5): war|=np.abs((pl[:,i]-pl[:,j]+180)%360-180)<=1
    F["Any two star-planets within 1 deg (BS 17 war)"]=(np.where(war,"yes","no"),["yes","no"],np.ones(n,bool))
    return F
def zscan(FE,FB):
    rows=[]
    for k,(ce,lv,ve) in FE.items():
        cb,_,vb=FB[k]; nE=ve.sum()
        for l in lv:
            if l=="no": continue            # binary features: only the "yes" bin (the "no" bin is its mirror)
            p=(cb[vb]==l).mean(); o=(ce[ve]==l).sum(); e=nE*p; z=(o-e)/np.sqrt(max(e*(1-p),1e-9)); rows.append((k,l,int(nE),int(o),e,z))
    return pd.DataFrame(rows,columns=["feature","bin","n","obs","exp","z"])
def prep(t,lat,lon,enr_ev,tb,ed,seed=11):
    rng=np.random.default_rng(seed); tb=(tb+pd.to_timedelta(rng.uniform(0,6*3600,len(tb)),unit='s')).reset_index(drop=True)
    enb=M.enrich(pd.DataFrame({'datetime':tb})); ix=rng.integers(0,len(t),len(tb))
    return features(t,lat,lon,enr_ev,ed),features(tb,lat[ix],lon[ix],enb,ed)
if __name__=='__main__':
    Q=pd.read_csv("quakes_m55_input.csv"); S=pd.read_csv("m55_features_sid.csv"); bg=pd.read_csv("bg_sid.csv")
    tq=pd.to_datetime(Q.datetime).dt.tz_localize(None).reset_index(drop=True); tb=pd.to_datetime(bg.datetime).dt.tz_localize(None).reset_index(drop=True)
    ed=np.array([E.dt_to_jd(e["date"]) for e in E.find(1850,2028)])
    sp=(tq>=pd.Timestamp("2008-01-01")).values; bsp=(tb>=pd.Timestamp("2008-01-01")).values
    out={}
    for lab,m,bm in (("disc",~sp,~bsp),("hold",sp,bsp)):
        FE,FB=prep(tq[m].reset_index(drop=True),Q.lat.values[m],Q.lon.values[m],S[m].reset_index(drop=True),tb[bm].reset_index(drop=True),ed)
        out[lab]=zscan(FE,FB)
    for lab,f,y0,y1 in (("older1","histA_declustered.csv",1964,1989),("older2","histB_declustered.csv",1900,1963)):
        Qh=pd.read_csv(f); th=pd.to_datetime(Qh.datetime).dt.tz_localize(None).reset_index(drop=True); Sh=M.enrich(Qh); G=grid(y0,y1)
        FE,FB=prep(th,Qh.latitude.values,Qh.longitude.values,Sh,G.datetime.reset_index(drop=True),ed); out[lab]=zscan(FE,FB)
    D=out["disc"].rename(columns={"n":"n_disc","obs":"obs_disc","exp":"exp_disc","z":"z_disc"})
    for lab in ("hold","older1","older2"):
        o=out[lab][["feature","bin","n","obs","exp","z"]].rename(columns={"n":f"n_{lab}","obs":f"obs_{lab}","exp":f"exp_{lab}","z":f"z_{lab}"}); D=D.merge(o,on=["feature","bin"])
    nb=len(D); bar=stats.norm.isf(0.05/nb/2)
    r=np.corrcoef(D.z_disc,D.z_hold)[0,1]
    # null for r: Pearson r of independent z's has sd ~1/sqrt(nb); report exact t-based p
    pr=stats.pearsonr(D.z_disc,D.z_hold)[1]
    D["p_disc"]=2*stats.norm.sf(D.z_disc.abs()); D["p_hold"]=2*stats.norm.sf(D.z_hold.abs()); D["p_older1"]=2*stats.norm.sf(D.z_older1.abs()); D["p_older2"]=2*stats.norm.sf(D.z_older2.abs())
    D["bar"]=0.05/nb
    D["disc_nominal"]=D.p_disc<0.05; D["replicates"]=(D.p_disc<0.05)&(D.p_hold<0.05)&(np.sign(D.z_disc)==np.sign(D.z_hold))
    D["passes_bonferroni_disc"]=D.p_disc<D.bar; D["passes_bonferroni_hold"]=D.p_hold<D.bar
    print("bins",nb,"Bonferroni bar p<",round(0.05/nb,5),"(|z|>",round(bar,2),")"); print("corr(z_disc,z_hold)",round(r,3),"p",round(pr,3))
    nd=int(D.disc_nominal.sum()); nrep=int(D.replicates.sum()); print("nominal discovery hits",nd,"expected",round(.05*nb,1),"replicated in holdout same direction",nrep,"expected by chance",round(nd*0.025,2))
    print("Bonferroni hits: disc",int(D.passes_bonferroni_disc.sum()),"hold",int(D.passes_bonferroni_hold.sum()),"older1",int((D.p_older1<D.bar).sum()),"older2",int((D.p_older2<D.bar).sum()))
    for nm in ("older1","older2"):
        rr=np.corrcoef(D.z_disc,D[f"z_{nm}"])[0,1]; print(nm,"corr with z_disc",round(rr,3),"p",round(stats.pearsonr(D.z_disc,D[f"z_{nm}"])[1],3),"nominal hits",int((D[f"p_{nm}"]<0.05).sum()))
    pd.set_option("display.width",260); pd.set_option("display.max_colwidth",60)
    print(D.reindex(D.z_disc.abs().sort_values(ascending=False).index).head(12)[["feature","bin","z_disc","z_hold","z_older1","z_older2"]].round(2).to_string())
    print(D[D.replicates][["feature","bin","z_disc","z_hold","z_older1","z_older2"]].round(2).to_string())
    D.to_csv("results/data_first_treatise_bin_scan.csv",index=False)

    if len(sys.argv)>1 and sys.argv[1]=='null':   # chance distribution for cross-era agreement (about 1 minute)
    
        Q=pd.read_csv("quakes_m55_input.csv"); S=pd.read_csv("m55_features_sid.csv"); bg=pd.read_csv("bg_sid.csv")
        tq=pd.to_datetime(Q.datetime).dt.tz_localize(None).reset_index(drop=True); tb=pd.to_datetime(bg.datetime).dt.tz_localize(None).reset_index(drop=True)
        ed=np.array([E.dt_to_jd(e["date"]) for e in E.find(1850,2028)])
        sp=(tq>=pd.Timestamp("2008-01-01")).values; bsp=(tb>=pd.Timestamp("2008-01-01")).values
        sets={}
        for lab,m,bm in (("disc",~sp,~bsp),("hold",sp,bsp)):
            FE,FB=prep(tq[m].reset_index(drop=True),Q.lat.values[m],Q.lon.values[m],S[m].reset_index(drop=True),tb[bm].reset_index(drop=True),ed); sets[lab]=(FE,FB,int(m.sum()))
        for lab,f,y0,y1 in (("older1","histA_declustered.csv",1964,1989),("older2","histB_declustered.csv",1900,1963)):
            Qh=pd.read_csv(f); th=pd.to_datetime(Qh.datetime).dt.tz_localize(None).reset_index(drop=True); Sh=M.enrich(Qh); G=grid(y0,y1)
            FE,FB=prep(th,Qh.latitude.values,Qh.longitude.values,Sh,G.datetime.reset_index(drop=True),ed); sets[lab]=(FE,FB,len(Qh))
        def mats(FB):
            cols=[];val=[];p=[]
            for k,(cb,lv,vb) in FB.items():
                for l in lv:
                    if l=="no": continue
                    cols.append((cb==l)&vb); val.append(vb); p.append((cb[vb]==l).mean())
            return np.array(cols).T.astype(np.int32),np.array(val).T.astype(np.int32),np.array(p)
        MT={k:mats(v[1]) for k,v in sets.items()}
        obs=zscan(sets["disc"][0],sets["disc"][1]).z.values
        def zdraw(lab,rng):
            I,V,p=MT[lab]; n=sets[lab][2]; idx=rng.integers(0,len(I),n); o=I[idx].sum(0); nv=V[idx].sum(0); e=nv*p; return (o-e)/np.sqrt(np.maximum(e*(1-p),1e-9))
        real={k:zscan(v[0],v[1]).z.values for k,v in sets.items()}
        rng=np.random.default_rng(17); N=3000; res={k:[] for k in ("hold","older1","older2")}
        for _ in range(N):
            zd=zdraw("disc",rng)
            for k in res: res[k].append(np.corrcoef(zd,zdraw(k,rng))[0,1])
        out=[]
        for k in res:
            r=np.corrcoef(real["disc"],real[k])[0,1]; a=np.array(res[k]); p=(np.abs(a)>=abs(r)).mean(); out.append((k,round(r,3),round(a.std(),3),p)); print(k,"observed r",round(r,3),"chance sd",round(a.std(),3),"two-sided p",p)
        pd.DataFrame(out,columns=["era_compared_with_1990_2007","observed_r","chance_sd","p_two_sided"]).to_csv("results/data_first_treatise_scan_agreement.csv",index=False)
