import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"../shared"); import myeph as M
exec(open(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),"bs_en.py")).read().split("def tmand")[0].split("from hist import grid")[1])
def tmand(h): return np.where(h<6,"Varuna",np.where(h<12,"Vayu",np.where(h<18,"Agni","Indra")))
ev=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv"); ev["datetime"]=ev.UTC
S=M.enrich(ev); n=nk(S.Moon_Lon.values); h=(pd.to_datetime(ev.UTC).dt.hour+pd.to_datetime(ev.UTC).dt.minute/60+ev.Lon/15)%24
T=tmand(h.values)
out=[]
for nm,m in (("Sanskrit",SAN),("English U.Ash in Indra",E1),("English U.Ash in Varuna",E2),("English, both set aside",E3)):
    c=[m.get(i,"unassigned") for i in n]; cnt={l:c.count(l) for l in ["Vayu","Agni","Indra","Varuna","unassigned"]}
    agree=sum(a==b for a,b in zip(c,T)); out.append((nm,cnt["Vayu"],cnt["Agni"],cnt["Indra"],cnt["Varuna"],cnt["unassigned"],agree))
df=pd.DataFrame(out,columns=["lists","Vayu","Agni","Indra","Varuna","unassigned","nakshatra circle = time circle"]); print(df.to_string()); print("time circles",{l:int((T==l).sum()) for l in ["Vayu","Agni","Indra","Varuna"]})
print(pd.DataFrame({"event":ev.Event,"nak":n,"San":[SAN[i] for i in n],"E1":[E1.get(i,"-") for i in n],"E2":[E2.get(i,"-") for i in n],"time":T,"mag":ev.Mag}).to_string())
df.to_csv("bs15_variants.csv",index=False)
