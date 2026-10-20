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
