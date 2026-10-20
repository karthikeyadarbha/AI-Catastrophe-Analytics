import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import eclipses as E
W=360/27
q=pd.read_csv("big.csv"); q=q[q.type=="earthquake"].copy(); q["t"]=pd.to_datetime(q.time).dt.tz_localize(None)
sid=pd.read_csv("quakes_m65_sid.csv").set_index("id"); trop=pd.read_csv("quakes_m65_trop.csv").set_index("id")
q=q[q.id.isin(sid.index)].reset_index(drop=True)
# ---- decluster (Gardner-Knopoff style)
q=q.sort_values("mag",ascending=False).reset_index(drop=True)
keep=np.ones(len(q),bool); lat=np.radians(q.latitude.values); lon=np.radians(q.longitude.values); tt=q.t.values.astype("datetime64[s]").astype(np.int64)/86400.0
for i in range(len(q)):
    if not keep[i]: continue
    M=q.mag[i]; T=10**(0.032*M+2.7389); R=10**(0.1238*M+0.983)
    dt=np.abs(tt-tt[i]); 
    d=6371*np.arccos(np.clip(np.sin(lat)*np.sin(lat[i])+np.cos(lat)*np.cos(lat[i])*np.cos(lon-lon[i]),-1,1))
    m=(dt<=T)&(d<=R); m[i]=False; m&=(q.mag.values<=M)
    keep[m]=False
Q=q[keep].sort_values("t").reset_index(drop=True); print("events",len(q),"declustered",len(Q))
MAND={}
for name,l in {"Anila":[14,22,0,11,12,13],"Anala":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]}.items():
    for k in l: MAND[k]=name
SHARP=[5,8,17,18]; SEAS=lambda sun,anchor:(((sun-anchor)%360)//60).astype(int)
def feats(df,frame):
    L={p:df[f"{p}_Lon"].values for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]}
    nk=lambda x:(x//W).astype(int)
    f=pd.DataFrame(index=df.index)
    f["mand"]=[MAND[k] for k in nk(L["Moon"])]
    f["sharp"]=np.isin(nk(L["Moon"]),SHARP)
    f["season"]=SEAS(L["Sun"],22*W if frame=="sid" else 270.0)
    f["jupsat"]=nk(L["Jupiter"])==nk(L["Saturn"])
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: f[p+"_R"]=df[p+"_Is_Retro"].astype(bool).values
    # frame-invariant
    el=(L["Moon"]-L["Sun"])%360; tith=(el//12).astype(int)+1; t15=(tith-1)%15+1
    f["rikta"]=np.isin(t15,[4,9,14]); f["syz"]=np.isin(tith,[29,30,1,14,15,16])
    pl=np.vstack([L[p] for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]]).T
    war=np.zeros(len(df),bool)
    for i in range(5):
        for j in range(i+1,5): war|=np.abs((pl[:,i]-pl[:,j]+180)%360-180)<=3
    f["war3"]=war
    oth=np.vstack([L[p] for p in ["Sun","Mars","Mercury","Jupiter","Venus","Saturn"]]).T
    f["moonpl5"]=(np.abs((L["Moon"][:,None]-oth+180)%360-180)<=5).any(1)
    return f
ev=E.find(1989,2028); ed=np.array([E.dt_to_jd(e["date"]) for e in ev])
def ecl(dtimes):
    jd=np.array([E.dt_to_jd(pd.Timestamp(x).to_pydatetime()) for x in dtimes]); return np.abs(jd[:,None]-ed[None,:]).min(1)
Qs=sid.loc[Q.id].reset_index(); Qt=trop.loc[Q.id].reset_index()
Fs=feats(Qs,"sid"); Ft=feats(Qt,"trop")
bs=pd.read_csv("bg_sid.csv"); bt=pd.read_csv("bg_trop.csv")
Bs=feats(bs,"sid"); Bt=feats(bt,"trop")
Bd=ecl(pd.to_datetime(bs.datetime).dt.tz_localize(None).values); Qd=ecl(Q.t.values)
split=(Q.t>=pd.Timestamp("2008-01-01")).values
PT={};BSm={}
for nm,l in {"Anila":[14,22,0,11,12,13],"Anala":[2,15,24,1,19,9,7,10],"Indra":[17,20,21,3,4,6,16],"Varuna":[23,25,26,5,8,18]}.items():
    for k in l: PT[k]=nm
for nm,l in {"Vayu":[11,12,13,14,6,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]}.items():
    for k in l: BSm[k]=nm
ptl=np.array([PT[k] for k in range(27)]); bsl=np.array([BSm[k] for k in range(27)])
SH=np.zeros(27,bool); SH[SHARP]=True
SYS=[("Lahiri (baseline)",0.0),("KP (23d21m, 1971)",-0.10),("Swamikannu Pillai",0.03),("Fagan-Bradley",0.88),("Raman",-1.44),("Sepharial/traditional (498 AD)",-1.85),("Chidambara Iyer",-3.2),("Tropical (ayanamsa 0)",23.85)]
def chi(ev,bg,lv):
    sh=np.array([(bg==l).mean() for l in lv]); o=np.array([(ev==l).sum() for l in lv]); n=o.sum(); return stats.chisquare(o,sh*n)[1]
def run(d):
    # ayanamsa differs by d from Lahiri -> sidereal longitude = Lahiri_lon - d
    nq=(((Qs.Moon_Lon.values-d)%360)//W).astype(int); nb=(((bs.Moon_Lon.values-d)%360)//W).astype(int)
    out=[]
    for lab,mask in (("disc",~split),("hold",split)):
        out.append(chi(ptl[nq][mask],ptl[nb],["Anila","Anala","Indra","Varuna"]))
        out.append(chi(bsl[nq][mask],bsl[nb],["Vayu","Agni","Indra","Varuna"]))
        p0=SH[nb].mean(); o=int(SH[nq][mask].sum()); out.append(stats.binom.sf(o-1,mask.sum(),p0))
    return out
rows=[]
for nm,d in SYS: rows.append((nm,d)+tuple(run(d)))
cols=["system","offset_deg_vs_Lahiri","PT_mand_disc_p","BS_mand_disc_p","sharp_disc_p","PT_mand_hold_p","BS_mand_hold_p","sharp_hold_p"]
R=pd.DataFrame(rows,columns=cols); pd.set_option("display.width",250); print(R.round(3).to_string()); R.to_csv("results/ayanamsa_named_systems_tests.csv",index=False)

for d in (0.0,0.88,1.25):
    nq=(((Qs.Moon_Lon.values-d)%360)//W).astype(int); nb=(((bs.Moon_Lon.values-d)%360)//W).astype(int); p0=SH[nb].mean()
    for lab,mask in (("disc",~split),("hold",split),("all",np.ones(len(split),bool))):
        n=mask.sum(); o=int(SH[nq][mask].sum()); print(d,lab,n,o,round(n*p0,1),round(o/(n*p0),2),round(stats.binom.sf(o-1,n,p0),4))
