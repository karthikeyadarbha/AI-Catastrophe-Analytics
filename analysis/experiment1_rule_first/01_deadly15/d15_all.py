import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
sys.path.insert(0,"../shared"); sys.path.insert(0,"../shared")
import myeph as M, eclipses as E
r=np.radians
d=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv")
IDX=[4,18,8,17,8,21,22,2,14,6,14,2,1,24,5]
def mk(x):
    o={}
    for n,l in x.items():
        for k in l:o[k]=n
    return o
BRH=mk({"Vayu":[11,12,13,14,4,0],"Agni":[7,2,15,1,9,24,10],"Indra":[17,16,21,22,6,3,20],"Varuna":[26,19,5,8,18,25,23]})
CMP=mk({"Vayu":[11,12,13,14,6,4,0],"Agni":[10,7,2,15,1,9],"Indra":[3,17,20,16,21,22],"Varuna":[26,19,5,8,18,25,23]})
t=pd.to_datetime(d.UTC); jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
lam=M.sun_trop(jd); dec=np.arcsin(np.sin(r(23.439))*np.sin(r(lam)))
h=(d.LocalSolarHour.values)%24
x=-np.tan(r(d.Lat.values))*np.tan(dec); ok=np.abs(x)<1; H0=np.degrees(np.arccos(np.clip(x,-1,1)))/15; sr,ss=12-H0,12+H0
LV=["Vayu","Agni","Indra","Varuna"]
def yama(same):
    out=[]
    for i in range(15):
        if not ok[i]: out.append("none (polar)"); continue
        day=ss[i]-sr[i]; night=24-day
        if sr[i]<=h[i]<ss[i]: out.append(LV[min(3,int((h[i]-sr[i])/(day/4)))])
        else: k=min(3,int(((h[i]-ss[i])%24)/(night/4))); out.append(LV[k] if same else LV[::-1][k])
    return out
ed=[E.dt_to_jd(e["date"]) for e in E.find(1998,2027)]; ed=np.array(ed)
ecl7=[bool(((jd[i]-ed)>=0).__and__((jd[i]-ed)<=7).any()) for i in range(15)]
out=pd.DataFrame({"Event":d.Event,"Mag":d.Mag,"Moon_Nakshatra":d.Moon_Nak,"PT_circle":d.PT_Mandala_Skt,"BS_circle":d.BS_Mandala,
 "AS_Brihaspati_circle":[BRH.get(i,"not listed") for i in IDX],"AS_compiler_circle":[CMP.get(i,"not listed") for i in IDX],
 "Local_solar_hour":d.LocalSolarHour,"BS_time_circle":d.Time_Mandala,"AS_yama_night_reversed":yama(False),"AS_yama_night_same":yama(True),
 "Eclipse_within_7d_before":ecl7})
out.to_csv("results/deadly15_circles_all_treatises.csv",index=False); print(out.to_string())
tw=lambda i: ok[i] and min(abs(h[i]-sr[i]),abs(h[i]-ss[i]))<=0.8
print("twilight",sum(tw(i) for i in range(15)),"ecl7",sum(ecl7))
print("yama match BS-list",sum(a==b for a,b in zip(out.AS_yama_night_reversed,out.BS_circle)),sum(a==b for a,b in zip(out.AS_yama_night_same,out.BS_circle)),"BRH",sum(a==b for a,b in zip(out.AS_yama_night_reversed,out.AS_Brihaspati_circle)),sum(a==b for a,b in zip(out.AS_yama_night_same,out.AS_Brihaspati_circle)))
