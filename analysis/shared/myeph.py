import numpy as np, pandas as pd
def jd_of(t): return 2440587.5+(pd.to_datetime(t)-pd.Timestamp("1970-01-01")).total_seconds()/86400 if not hasattr(t,'dt') else 2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds()/86400
R=np.radians
def ayan(jd): return 23.853+0.01397*((jd-2451545.0)/365.25)
def sun_trop(jd):
    T=(jd-2451545.0)/36525; L0=280.46646+36000.76983*T; M=R(357.52911+35999.05029*T)
    C=(1.914602-0.004817*T)*np.sin(M)+0.019993*np.sin(2*M)+0.000289*np.sin(3*M); return (L0+C-0.00569)%360
MT=[(0,0,1,0,6288774),(2,0,-1,0,1274027),(2,0,0,0,658314),(0,0,2,0,213618),(0,1,0,0,-185116),(0,0,0,2,-114332),(2,0,-2,0,58793),(2,-1,-1,0,57066),(2,0,1,0,53322),(2,-1,0,0,45758),(0,1,-1,0,-40923),(1,0,0,0,-34720),(0,1,1,0,-30383),(2,0,0,-2,15327),(0,0,1,2,-12528),(0,0,1,-2,10980),(4,0,-1,0,10675),(0,0,3,0,10034),(4,0,-2,0,8548),(2,1,-1,0,-7888),(2,1,0,0,-6766),(1,0,-1,0,-5163),(1,1,0,0,4987),(2,-1,1,0,4036),(2,0,2,0,3994),(4,0,0,0,3861),(2,0,-3,0,3665),(0,1,-2,0,-2689),(2,0,-1,2,-2602),(2,-1,-2,0,2390),(1,0,1,0,-2348),(2,-2,0,0,2236),(0,1,2,0,-2120),(0,2,0,0,-2069),(2,-2,-1,0,2048),(2,0,1,-2,-1773),(2,0,0,2,-1595),(4,-1,-1,0,1215),(0,0,2,2,-1110),(3,0,-1,0,-892),(2,1,1,0,-810),(4,-1,-2,0,759),(0,2,-1,0,-713),(2,2,-1,0,-700),(2,1,-2,0,691),(2,-1,0,-2,596),(4,0,1,0,549),(0,0,4,0,537),(4,-1,0,0,520),(1,0,-2,0,-487),(2,1,0,-2,-399),(0,0,2,-2,-381),(1,1,1,0,351),(3,0,-2,0,-340),(4,0,-3,0,330),(2,-1,2,0,327),(0,2,1,0,-323),(1,1,-1,0,299),(2,0,3,0,294)]
def moon_trop(jd):
    T=(jd-2451545.0)/36525
    Lp=218.3164477+481267.88123421*T; D=R(297.8501921+445267.1114034*T); M=R(357.5291092+35999.0502909*T); Mp=R(134.9633964+477198.8675055*T); F=R(93.2720950+483202.0175233*T)
    E=1-0.002516*T; s=np.zeros_like(T)
    for d,m,mp,f,c in MT: s+=c*E**abs(m)*np.sin(d*D+m*M+mp*Mp+f*F)
    A1=R(119.75+131.849*T); A2=R(53.09+479264.290*T)
    s+=3958*np.sin(A1)+1962*np.sin(R(Lp)-F)+318*np.sin(A2)
    return (Lp+s/1e6)%360
EL={ # a,da,e,de,I,dI,L,dL,w,dw,O,dO
"Mercury":(0.38709927,0.00000037,0.20563593,0.00001906,7.00497902,-0.00594749,252.25032350,149472.67411175,77.45779628,0.16047689,48.33076593,-0.12534081),
"Venus":(0.72333566,0.00000390,0.00677672,-0.00004107,3.39467605,-0.00078890,181.97909950,58517.81538729,131.60246718,0.00268329,76.67984255,-0.27769418),
"Earth":(1.00000261,0.00000562,0.01671123,-0.00004392,-0.00001531,-0.01294668,100.46457166,35999.37244981,102.93768193,0.32327364,0.0,0.0),
"Mars":(1.52371034,0.00001847,0.09339410,0.00007882,1.84969142,-0.00813131,-4.55343205,19140.30268499,-23.94362959,0.44441088,49.55953891,-0.29257343),
"Jupiter":(5.20288700,-0.00011607,0.04838624,-0.00013253,1.30439695,-0.00183714,34.39644051,3034.74612775,14.72847983,0.21252668,100.47390909,0.20469106),
"Saturn":(9.53667594,-0.00125060,0.05386179,-0.00050991,2.48599187,0.00193609,49.95424423,1222.49362201,92.59887831,-0.41897216,113.66242448,-0.28867794)}
def helio(name,T):
    a0,da,e0,de,I0,dI,L0,dL,w0,dw,O0,dO=EL[name]
    a=a0+da*T;e=e0+de*T;I=R(I0+dI*T);L=L0+dL*T;w=w0+dw*T;O=R(O0+dO*T)
    Mn=R((L-w+180)%360-180); E=Mn+e*np.sin(Mn)
    for _ in range(8): E=E-(E-e*np.sin(E)-Mn)/(1-e*np.cos(E))
    x=a*(np.cos(E)-e); y=a*np.sqrt(1-e*e)*np.sin(E); wp=R(w)-O
    cw,sw,cO,sO,ci,si=np.cos(wp),np.sin(wp),np.cos(O),np.sin(O),np.cos(I),np.sin(I)
    X=(cw*cO-sw*sO*ci)*x+(-sw*cO-cw*sO*ci)*y; Y=(cw*sO+sw*cO*ci)*x+(-sw*sO+cw*cO*ci)*y; Z=(sw*si)*x+(cw*si)*y
    return X,Y,Z
def geo_trop(name,jd):
    T=(jd-2451545.0)/36525; X,Y,Z=helio(name,T); Xe,Ye,Ze=helio("Earth",T)
    lon=np.degrees(np.arctan2(Y-Ye,X-Xe))%360
    return (lon+1.396971*T)%360   # precess J2000 -> date
def all_trop(jd):
    out={"Sun":sun_trop(jd),"Moon":moon_trop(jd)}
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]:
        l=geo_trop(p,jd); l2=geo_trop(p,jd+0.5); l1=geo_trop(p,jd-0.5)
        out[p]=l; out[p+"_retro"]=(((l2-l1+180)%360)-180)<0
    return out
def node_trop(jd):
    T=(jd-2451545.0)/36525; return (125.0445479-1934.1362891*T+0.0020754*T*T)%360   # mean ascending node (Rahu)
def speed(name,jd):
    return ((geo_trop(name,jd+0.5)-geo_trop(name,jd-0.5)+180)%360)-180
def enrich(df,tcol="datetime"):
    t=pd.to_datetime(df[tcol]).dt.tz_localize(None); jd=2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
    A=all_trop(jd); ay=ayan(jd); o=pd.DataFrame(index=df.index)
    for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]: o[p+"_Lon"]=(A[p]-ay)%360
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: o[p+"_Is_Retro"]=A[p+"_retro"]
    return o
import os as _os, sys as _sys
SKY_ENGINE, SKY_ENGINE_VERSION = "modern", None
if _os.environ.get("SKY_ENGINE", "modern").lower() != "modern":   # Indian-method sky, see indic_sky.py
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); import indic_sky as _isky; _isky.install(globals(), _os.environ["SKY_ENGINE"])
if __name__=="__main__":
    q=pd.read_csv("quakes_m65_sid.csv"); o=enrich(q)
    for p in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]:
        d=((o[p+"_Lon"]-q[p+"_Lon"]+180)%360)-180; print(p,"median |diff|",round(d.abs().median(),3),"p99",round(d.abs().quantile(.99),3),"max",round(d.abs().max(),3),"bias",round(d.mean(),3))
    for p in ["Mars","Mercury","Jupiter","Venus","Saturn"]: print(p,"retro agree",round((o[p+"_Is_Retro"].astype(bool)==q[p+"_Is_Retro"].astype(bool)).mean(),4))
