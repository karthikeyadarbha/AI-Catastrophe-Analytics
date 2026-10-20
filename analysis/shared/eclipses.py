"""Eclipse finder using Meeus, Astronomical Algorithms ch.54 (mean-phase JDE, no periodic-term time corrections).
Existence + magnitude from gamma/u; date accurate to about +/-0.6 day (adequate for window tests)."""
import math
from datetime import datetime, timedelta
def jd_to_dt(jd):
    return datetime(2000,1,1,12)+timedelta(days=jd-2451545.0)
def dt_to_jd(dt):
    return 2451545.0+(dt-datetime(2000,1,1,12)).total_seconds()/86400.0
def _phase(k):
    T=k/1236.85
    jde=2451550.09766+29.530588861*k+0.00015437*T**2-0.000000150*T**3+0.00000000073*T**4
    E=1-0.002516*T-0.0000074*T**2
    M=math.radians((2.5534+29.10535670*k-0.0000014*T**2-0.00000011*T**3)%360)
    Mp=math.radians((201.5643+385.81693528*k+0.0107582*T**2+0.00001238*T**3-0.000000058*T**4)%360)
    F=(160.7108+390.67050284*k-0.0016118*T**2-0.00000227*T**3+0.000000011*T**4)%360
    Om=math.radians((124.7746-1.56375588*k+0.0020672*T**2+0.00000215*T**3)%360)
    F1=math.radians(F-0.02665*math.sin(Om))
    P=(0.2070*E*math.sin(M)+0.0024*E*math.sin(2*M)-0.0392*math.sin(Mp)+0.0116*math.sin(2*Mp)
       -0.0073*E*math.sin(M+Mp)+0.0067*E*math.sin(M-Mp)+0.0118*math.sin(2*F1))
    Q=(5.2207-0.0048*E*math.cos(M)+0.0020*E*math.cos(2*M)-0.3299*math.cos(Mp)
       -0.0060*E*math.cos(M+Mp)+0.0041*E*math.cos(M-Mp))
    W=abs(math.cos(F1))
    gam=(P*math.cos(F1)+Q*math.sin(F1))*(1-0.0048*W)
    u=(0.0059+0.0046*E*math.cos(M)-0.0182*math.cos(Mp)+0.0004*math.cos(2*Mp)-0.0005*math.cos(M+Mp))
    return jde,gam,u
def find(y0,y1):
    out=[]
    k0=math.floor((y0-2000)*12.3685)-2; k1=math.ceil((y1-2000)*12.3685)+2
    k=k0
    while k<=k1:
        for half in (0.0,0.5):
            kk=k+half
            if abs(kk-round(kk))<1e-9: kind="solar"
            else: kind="lunar"
            jde,gam,u=_phase(kk)
            if kind=="solar":
                if abs(gam)<1.5433+u:
                    out.append(dict(date=jd_to_dt(jde),kind="solar",gamma=gam,mag=None))
            else:
                pen=(1.5573+u-abs(gam))/0.5450
                um=(1.0128-u-abs(gam))/0.5450
                if pen>0:
                    out.append(dict(date=jd_to_dt(jde),kind="lunar",gamma=gam,umbral=um>0,mag=um if um>0 else pen))
        k+=1
    out.sort(key=lambda d:d["date"])
    return out
