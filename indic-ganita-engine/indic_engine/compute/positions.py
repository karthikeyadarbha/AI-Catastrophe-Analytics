"""Mean and true positions by the methods of the Tantrasangraha (Chapters 1 and 2), with the Aryabhatiya parameter set as an alternative
and the Madhava-Newton series available for the Rsine. All longitudes are nirayana (Mesadi fixed to the stars), degrees.
Rule IDs in comments refer to rules/data/*.json."""
import math
from . import params as P
from .jya import jya, kotijya, capa, trijya
D2M=60.0*1.0  # degrees -> minutes
def _m(deg): return deg*60.0
def _d(minutes): return minutes/60.0
def ahargana(jd_ut):
    """continuous days since the Kali epoch measured to mean sunrise at Ujjayini (TS 1.11-1.12; epoch fixed by the book's worked example)"""
    off=(P.SUNRISE_LOCAL_H-P.UJJAIN_LON/15.0)/24.0
    return jd_ut-P.KALI_EPOCH_JD_MIDNIGHT-off
def mean_longitudes(jd_ut,param="TS"):
    """TS-MEAN-1.12: longitude = dhruva + 360 * A * N / civil_days (fractional revolutions). Node moves backwards (TS 1.15 v.37)."""
    A=ahargana(jd_ut); out={}
    for k,N in P.REV[param].items():
        f=(A*N/P.CIVIL_DAYS_MAHAYUGA)%1.0
        if k=="MoonNode": out[k]=((1.0-f)*360.0+P.DHRUVA[param][k])%360.0
        else: out[k]=(f*360.0+P.DHRUVA[param][k])%360.0
    return out
def _sn(deg,s): return jya(_m(deg),s)/trijya(s)                       # signed sine, unit radius
def _arc(x,s): return _d(capa(x*trijya(s),s))                          # degrees from a signed unit sine
def true_sun(mean_sun,s="madhava"):
    """TS-SUN-2.10: true = mean - arcsin( (3/80) sin(mean - 78 deg) )"""
    return (mean_sun-_arc(3/80*_sn(mean_sun-P.MANDOCCA["Sun"],s),s))%360
def true_moon(mean_moon,apogee,s="madhava"):
    """TS-MOON-2.14: true = mean - arcsin( (7/80) sin(mean - apogee) )"""
    return (mean_moon-_arc(7/80*_sn(mean_moon-apogee,s),s))%360
def _rho(planet,sinabs,s):
    if planet=="Saturn": return 39/320                    # TS-MANDA-SAT 2.26 v.61 (1/8 less 1/40 of it, eq 2.170)
    if planet=="Mars": return (7+sinabs)/39               # v.61-62
    if planet=="Jupiter": return (7+sinabs)/82
    if planet=="Mercury": return 1/6                      # v.68-69
    if planet=="Venus": return 1/(14+trijya(s)*sinabs/240)  # v.75
def _manda_arc(planet,kappa,s):
    sk=_sn(kappa,s); return _arc(_rho(planet,abs(sk),s)*sk,s)   # signed degrees; true = mean - arc
def _avisista_karna(planet,kappa,s,iters=60):
    """TS-KARNA-2.17: iterate K from the epicycle radius r0 = rho*R until it stops changing"""
    R=trijya(s); sk=_sn(kappa,s); ck=_sn(kappa+90.0,s); r0=_rho(planet,abs(sk),s)*R
    d1,k1=r0*sk,r0*ck; K=math.hypot(R+k1,d1)
    for _ in range(iters):
        d,k=K*d1/R,K*k1/R; K2=math.hypot(R+k,d)
        if abs(K2-K)<1e-9: K=K2; break
        K=K2
    return K
_SIG={"Mars":(53,2),"Jupiter":(16,1),"Saturn":(9,1),"Mercury":(31,2),"Venus":(59,2)}
def _sighra_exterior(planet,theta_ms,theta_sun,s,frac=1.0):
    """TS-SIGHRA-EXT 2.26 vv.63-67: returns arc to subtract from the manda-sphuta (K replaced by trijya)"""
    R=trijya(s); sg=_sn(theta_ms-theta_sun,s); c,k=_SIG[planet]; rs=R*(c-k*abs(sg))/80
    Ks=math.hypot(R+rs*_sn(theta_ms-theta_sun+90,s),rs*sg); return frac*_arc(rs*sg/Ks,s)
def true_exterior(planet,mean,mean_sun,s="madhava"):
    """TS-EXT-4STEP 2.26 vv.60-67: half manda, half sighra, full manda on the mean planet, full sighra"""
    th=P.MANDOCCA[planet]
    P1=mean-0.5*_manda_arc(planet,mean-th,s)
    P2=P1-_sighra_exterior(planet,P1,mean_sun,s,0.5)
    P3=mean-_manda_arc(planet,P2-th,s)
    return (P3-_sighra_exterior(planet,P3,mean_sun,s,1.0))%360
def true_interior(planet,mean,mean_sun,s="madhava"):
    """TS-INT-2STEP 2.27-2.28 vv.68-79: manda-sphuta of the heliocentric planet, then sighra on the mean Sun (the sighrocca)"""
    R=trijya(s); th=P.MANDOCCA[planet]; kap=mean-th
    theta_ms=mean-_manda_arc(planet,kap,s); K=_avisista_karna(planet,kap,s)
    sg=_sn(theta_ms-mean_sun,s); c,k=_SIG[planet]; guna=c-k*abs(sg); sphutaguna=guna*K/R
    dohphala=R*sg*sphutaguna/80; kotiphala=R*_sn(theta_ms-mean_sun+90,s)*sphutaguna/80
    Ks=math.hypot(R+kotiphala,dohphala); return (mean_sun+_arc(dohphala/Ks,s))%360
def planet_positions(jd_ut,param="TS",scheme="madhava"):
    """nirayana true longitudes of Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn plus the lunar node and apogee, and the mean values"""
    m=mean_longitudes(jd_ut,param); ms=m["Sun"]; t={}
    t["Sun"]=true_sun(ms,scheme); t["Moon"]=true_moon(m["Moon"],m["MoonApogee"],scheme)
    for p in ("Mars","Jupiter","Saturn"): t[p]=true_exterior(p,m[p],ms,scheme)
    for p in ("Mercury","Venus"): t[p]=true_interior(p,m[p],ms,scheme)
    t["MoonNode"]=m["MoonNode"]; t["MoonApogee"]=m["MoonApogee"]
    return {"true":t,"mean":m,"ahargana":ahargana(jd_ut),"param":param,"scheme":scheme}
