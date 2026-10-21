"""Surya-siddhanta (SS) mean and true positions, read from the Hindi-commentary edition (Vijnana-bhashya of Pt. Mahavir Prasad Srivastava).
Rule IDs SS.* refer to rules/data/suryasiddhanta.json. Longitudes are nirayana degrees in the SS frame
(mean midnight at Ujjayini/Lanka, Kali epoch JD 588465.5 = start of Kali, Friday 18 Feb 3102 BCE)."""
import math
from . import params as P
YUGA_YEARS=4_320_000
CIVIL_DAYS=1_577_917_828                                   # SS 1.37 civil days per mahayuga
REV={ # SS 1.29-33: revolutions per mahayuga (Mercury and Venus: sighra revolutions; their 'mean' is the Sun)
 "Sun":4_320_000,"Moon":57_753_336,"Mars":2_296_832,"Mercury":17_937_060,"Jupiter":364_220,"Venus":7_022_376,"Saturn":146_568,
 "MoonApogee":488_203,"MoonNode":-232_238}
KALPA_REV={ # SS 1.41-44: revolutions per kalpa (1000 mahayugas) of the slow manda apsides and nodes
 "Sun":387,"Mars":204,"Mercury":368,"Jupiter":900,"Venus":535,"Saturn":39}
NODE_KALPA={"Mars":-214,"Mercury":-488,"Jupiter":-174,"Venus":-903,"Saturn":-662}   # retrograde
YEARS_SINCE_CREATION_AT_KALI=1_955_880_000                    # SS 1.45-47: Sun-years from creation to the start of Kali (after the 17,064,000 years of creation)
YEAR_LEN=CIVIL_DAYS/YUGA_YEARS                                # 365.258756481 days
# SS 2.34-37: manda and sighra epicycle circumference in degrees (of 360): at the end of even padas (anomaly 0, 180) and odd padas (90, 270)
MANDA_EVEN={"Sun":14.0,"Moon":32.0,"Mars":75.0,"Mercury":30.0,"Jupiter":33.0,"Venus":12.0,"Saturn":49.0}
MANDA_ODD ={"Sun":13+40/60,"Moon":31+40/60,"Mars":72.0,"Mercury":28.0,"Jupiter":32.0,"Venus":11.0,"Saturn":48.0}
SIGHRA_EVEN={"Mars":235.0,"Mercury":133.0,"Jupiter":70.0,"Venus":262.0,"Saturn":39.0}
SIGHRA_ODD ={"Mars":232.0,"Mercury":132.0,"Jupiter":72.0,"Venus":260.0,"Saturn":40.0}
# SS 2.17-22: the 24 Rsine differences (minutes) for R = 3438 at 3deg45' steps
RSINE_DIFF=[225,224,222,219,215,210,205,199,191,183,174,164,154,143,131,119,106,93,79,65,51,37,22,7]
R=3438.0
RS=[0]; 
for _d in RSINE_DIFF: RS.append(RS[-1]+_d)
def apsis_start(kalpa_rev):
    """longitude of an apsis or node at the start of Kali: kalpa revolutions * (years since creation / kalpa years); worked example SS 2.ch.2 (Vijnana-bhashya)"""
    return ((kalpa_rev*YEARS_SINCE_CREATION_AT_KALI/(YUGA_YEARS*1000.0))%1.0)*360.0
def rsin(deg,exact=False):
    """Rsine for R=3438. Default: SS rule 2.31-32, linear interpolation in the 24-entry table; exact=True uses math.sin"""
    a=abs(deg)%360.0; s=1.0
    if a>180: a-=180; s=-1.0
    if a>90: a=180-a
    if exact is True: v=R*math.sin(math.radians(a))
    else:
        x=a/3.75; i=min(int(x),23); v=RS[i]+(x-i)*(RS[i+1]-RS[i])
    sgn=(1.0 if (deg%360.0)<=180 else -1.0)
    return sgn*v
def rcos(deg,exact=False): return rsin(deg+90.0,exact)
def rarcsin(v,exact=False):
    """arc (degrees) with Rsine v: SS 2.33 inverse interpolation. exact='book' reproduces the commentator's worked-example shortcut (arc in minutes taken equal to the Rsine value)"""
    s=1.0 if v>=0 else -1.0; v=abs(v)
    if exact=="book": return s*v/60.0
    if exact: return s*math.degrees(math.asin(min(1.0,v/R)))
    i=max(j for j in range(24) if RS[j]<=v) if v<R else 23
    return s*3.75*(i+(v-RS[i])/(RS[i+1]-RS[i]))
def manda_circumference(b,kendra,exact=False):
    """SS 2.38: corrected circumference = even-end value + (odd-end - even-end) * |Rsin(kendra)| / R"""
    e,o=MANDA_EVEN[b],MANDA_ODD[b]; return e+(o-e)*abs(rsin(kendra,exact))/R
def sighra_circumference(b,kendra,exact=False):
    e,o=SIGHRA_EVEN[b],SIGHRA_ODD[b]; return e+(o-e)*abs(rsin(kendra,exact))/R
def manda_phala(b,kendra,exact=False):
    """SS 2.39: bhujaphala = circ*Rsin(kendra)/360; manda phala = arc whose Rsine is bhujaphala. Signed: kendra = mean - apogee, true = mean - phala"""
    bf=manda_circumference(b,kendra,exact)*rsin(kendra,exact)/360.0
    return rarcsin(bf,exact)
def sighra_phala(b,kendra,exact=False):
    """SS 2.40-42: karna = sqrt((R +- kotiphala)^2 + bhujaphala^2); phala = arc of Rsine R*bhujaphala/karna. kendra = sighrocca - planet; true = planet + phala"""
    c=sighra_circumference(b,kendra,exact)/360.0
    bf=c*rsin(kendra,exact); kf=c*rcos(kendra,exact)
    karna=math.hypot(R+kf,bf)
    return rarcsin(R*bf/karna,exact)
def ahargana(jd_ut):
    """continuous days since the Kali epoch, measured to mean midnight at Ujjayini (SS 1.46-50 midnight day-count)"""
    return jd_ut+P.UJJAIN_LON/360.0-P.KALI_EPOCH_JD_MIDNIGHT
def mean_longitudes(jd_ut,D=None):
    """SS.mean: longitude = 360*frac(D*rev/CIVIL_DAYS); planets at 0 at Kali start; Moon apogee 90 deg and node 180 deg (SS 1.57-58 + creation count)"""
    D=ahargana(jd_ut) if D is None else D; out={}
    for k,N in REV.items():
        f=(D*N/CIVIL_DAYS)%1.0
        if k=="MoonApogee": out[k]=(90.0+f*360.0)%360.0
        elif k=="MoonNode": out[k]=(180.0+f*360.0)%360.0
        else: out[k]=(f*360.0)%360.0
    return out
def apsides(jd_ut,D=None):
    """SS.apsides: manda apogees (mandocca) and nodes of the five planets, from kalpa revolutions and the creation-year count (SS 1.41-44, 2.ch worked example)"""
    D=ahargana(jd_ut) if D is None else D
    yrs=YEARS_SINCE_CREATION_AT_KALI+D/YEAR_LEN
    out={}
    for b,k in KALPA_REV.items(): out[b]=((k*yrs/(YUGA_YEARS*1000.0))%1.0)*360.0
    nodes={b:((k*yrs/(YUGA_YEARS*1000.0))%1.0)*360.0 for b,k in NODE_KALPA.items()}
    return out,nodes
def true_sun(m,ap,exact=False): return (m["Sun"]-manda_phala("Sun",m["Sun"]-ap["Sun"],exact))%360
def true_moon(m,exact=False): return (m["Moon"]-manda_phala("Moon",m["Moon"]-m["MoonApogee"],exact))%360
def true_planet(b,m,ap,exact=False):
    """SS 2.43-44 (four samskaras): half sighra on the mean, half manda on that, full manda (from the mean) and full sighra.
    Mars, Jupiter, Saturn: sighrocca = mean Sun, mean planet = own mean. Mercury, Venus: mean planet = mean Sun, sighrocca = own sighra mean."""
    if b in ("Mercury","Venus"): mean,soc=m["Sun"],m[b]
    else: mean,soc=m[b],m["Sun"]
    u=ap[b]
    p1=mean+0.5*sighra_phala(b,soc-mean,exact)
    mp1=manda_phala(b,p1-u,exact)
    p2=p1-0.5*mp1
    mp2=manda_phala(b,p2-u,exact)
    ms=mean-mp2
    return (ms+sighra_phala(b,soc-ms,exact))%360
# SS 2.52-55: sighra anomaly at which retrograde motion starts (it ends at 360 minus this value)
VAKRA_START={"Mars":164.0,"Mercury":144.0,"Jupiter":130.0,"Venus":163.0,"Saturn":115.0}
def is_retrograde(b,jd_ut,exact=False):
    """SS.vakra: retrograde while the final sighra anomaly lies between the station value and 360 minus it"""
    m=mean_longitudes(jd_ut); ap,_=apsides(jd_ut)
    if b in ("Mercury","Venus"): mean,soc=m["Sun"],m[b]
    else: mean,soc=m[b],m["Sun"]
    u=ap[b]
    p1=mean+0.5*sighra_phala(b,soc-mean,exact); p2=p1-0.5*manda_phala(b,p1-u,exact)
    ms=mean-manda_phala(b,p2-u,exact); k=(soc-ms)%360.0
    return VAKRA_START[b]<=k<=360.0-VAKRA_START[b]
def planet_positions(jd_ut,exact=False):
    m=mean_longitudes(jd_ut); ap,nodes=apsides(jd_ut)
    t={"Sun":true_sun(m,ap,exact),"Moon":true_moon(m,exact)}
    for b in ("Mars","Mercury","Jupiter","Venus","Saturn"): t[b]=true_planet(b,m,ap,exact)
    t["MoonNode"]=m["MoonNode"]; t["MoonApogee"]=m["MoonApogee"]
    return {"true":t,"mean":m,"mandocca":ap,"planet_nodes":nodes,"ahargana":ahargana(jd_ut),"model":"Suryasiddhanta","exact_sines":exact}
