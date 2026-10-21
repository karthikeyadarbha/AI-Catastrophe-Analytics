"""Drk (observation-corrected) computation of Paramesvara's Drgganita (Saka 1353 = 1431 CE), K.V. Sarma's edition (Hoshiarpur 1963).
Everything here is read from the printed numerals in the edition. Rule IDs DG.* refer to rules/data/drgganita.json.
All longitudes are nirayana degrees in the Drgganita's own frame (mean sunrise at Ujjayini, Kali epoch as in positions.ahargana)."""
import math
from .positions import ahargana
# Part II vv.2-11 (= Part I ch.1 vv.10-23): mean revolutions per day = G/H; the fractional part of D*G/H is the mean longitude.
# Then a correction of 1 unit per N days (unit minute or second of arc; sign +1 add / -1 subtract), then the dhruva (epoch value).
MEAN={ # body: (G, H, N, unit_in_degrees, sign, dhruva_degrees)   DG.mean.*
 "Sun":        (116,   42370,  45486,  1/60,   -1, 20/60+22/3600),
 "Moon":       (143,   3907,   8310,   1/60,   +1, 3+15/60+2/3600),
 "Mars":       (1,     687,    8719,   1/60,   +1, 3/60),
 "Mercury":    (33,    2903,   67452,  1/60,   -1, -(3+44/60)),
 "Jupiter":    (53,    229613, 141478, 1/3600, +1, -(2+59/60)),
 "Venus":      (73,    16403,  37348,  1/60,   +1, -4.0),
 "Saturn":     (13,    139955, 1395953,1/60,   -1, 4+8/60),
 "MoonApogee": (74,    239171, 47036,  1/3600, -1, 85+44/60+26/3600),
 "MoonNode":   (29,    197041, 92920,  1/3600, +1, 182+36/60),
}
MANDOCCA={"Sun":78.0,"Mars":127.0,"Mercury":220.0,"Jupiter":172.0,"Venus":80.0,"Saturn":236.0}   # DG.apsides, Part I ch.3 vv.1-2 and Part II vv.16-17
# Part II vv.26-42: tables of the equation (arc-minutes) at bhuja 6,12,...,90 degrees (15 entries). Entry 0 at bhuja 0 is zero.
MANDA={
 "Mars":   [68,133,200,266,330,392,452,507,557,601,639,669,689,704,707],
 "Mercury":[23,46,69,90,110,129,146,162,175,187,198,205,210,214,215],
 "Jupiter":[32,64,96,127,158,187,215,241,265,285,303,316,326,331,333],
 "Venus":  [18,33,48,55,60,66,71,75,78,81,83,84,85,85,86],
 "Saturn": [43,85,127,168,207,244,278,309,337,362,383,398,410,417,420],
 "Sun":    [13,27,40,52,64,76,86,96,104,112,118,123,126,128,129],
 "Moon":   [31,63,93,122,150,177,201,224,243,261,275,286,294,299,301],
}
# sighra tables: (makaradi, karkyadi). makaradi is used when the sighra anomaly lies in Makara..Mithuna (270..90), karkyadi in 90..270.
SIGHRA={
 "Mars":   ([143,285,426,566,706,843,978,1113,1245,1372,1498,1622,1738,1851,1958],
            [680,1256,1697,2001,2201,2328,2388,2414,2401,2364,2308,2236,2154,2059,1958]),
 "Mercury":([101,199,296,390,483,573,660,744,824,900,971,1037,1097,1151,1197],
            [223,435,624,791,931,1044,1134,1200,1245,1274,1283,1276,1262,1233,1197]),
 "Jupiter":([59,118,176,231,284,335,384,429,472,511,546,576,601,621,637],
            [89,175,256,330,398,458,510,553,589,615,634,646,649,646,637]),
 "Venus":  ([152,303,454,604,753,901,1051,1197,1341,1483,1623,1761,1893,2024,2150],
            [927,1657,2162,2466,2645,2755,2783,2788,2734,2667,2579,2483,2383,2269,2150]),   # 5th karkyadi entry printed 2445: read as 2645 (smooth differences); flagged DG.table.venus
 "Saturn": ([36,70,104,135,164,192,218,242,263,282,299,313,325,332,337],               # 14th printed 337, editor's note says 332
            [44,87,127,163,197,227,255,277,297,313,325,333,338,338,337]),
}
def _interp(tab,b):
    """second-order (Newton) interpolation in the 6-degree table; b = bhuja in degrees, 0..90; returns minutes"""
    f=[0.0]+list(tab)
    x=b/6.0; i=min(int(x),len(f)-3); t=x-i
    d1=f[i+1]-f[i]; d2=f[i+2]-2*f[i+1]+f[i]
    return f[i]+t*d1+t*(t-1)/2*d2
def _bhuja(k):
    k%=360.0
    if k<=90: return k
    if k<=180: return 180-k
    if k<=270: return k-180
    return 360-k
def manda_phala(body,kendra):
    """signed equation in degrees: positive when the anomaly (mean - apogee) is in 0..180"""
    v=_interp(MANDA[body],_bhuja(kendra))/60.0
    return v if (kendra%360)<=180 else -v
def sighra_phala(body,kendra):
    """signed sighra equation in degrees, kendra = sighrocca - planet (exterior) or planet(own mean) - mean Sun (interior)"""
    k=kendra%360.0; mk,kk=SIGHRA[body]
    tab=mk if (k<=90 or k>=270) else kk
    v=_interp(tab,_bhuja(k))/60.0
    return v if k<=180 else -v
def mean_longitudes(jd_ut):
    D=ahargana(jd_ut); out={}
    for b,(G,H,N,u,s,dh) in MEAN.items():
        rev=((D*G/H)%1.0)*360.0
        v=rev+s*(D/N)*u+dh
        out[b]=(360.0-v)%360.0 if b=="MoonNode" else v%360.0      # Rahu is "cakrat suddhah" (Part I ch.3 v.8)
    return out
def true_sun(m): return (m["Sun"]-manda_phala("Sun",m["Sun"]-MANDOCCA["Sun"]))%360
def true_moon(m): return (m["Moon"]-manda_phala("Moon",m["Moon"]-m["MoonApogee"]))%360
def true_exterior(b,m):
    """DG.4step (Part I ch.3 vv.18-19, identical in order to TS 2.26): half manda, half sighra, full manda on the mean, full sighra"""
    mean=m[b]; u=MANDOCCA[b]; S=m["Sun"]
    p1=mean-0.5*manda_phala(b,mean-u)
    p2=p1+0.5*sighra_phala(b,S-p1)
    p3=mean-manda_phala(b,p2-u)
    return (p3+sighra_phala(b,S-p3))%360
def true_interior(b,m,variant="H"):
    own=m[b]; S=m["Sun"]; u=MANDOCCA[b]
    if variant=="H":   # DG.interior: Part I ch.3 v.22 says Budha and Sukra work on the sighra circle, so the order is reversed: sighra first (elongation from the mean Sun), then manda
        x=S+sighra_phala(b,own-S)
        return (x-manda_phala(b,x-u))%360
    if variant=="A":   # manda on the own mean (heliocentric), sighra elongation on the mean Sun (TS-like order; poorer fit)
        th=own-manda_phala(b,own-u)
        return (S+sighra_phala(b,th-S))%360
    # variant B: literal reading of Part I ch.3 vv.20-21 (sighra first, then manda), start = mean Sun, sighrocca = own mean
    x1=S+0.5*sighra_phala(b,own-S)
    x2=x1-0.5*manda_phala(b,x1-u)
    x3=S+sighra_phala(b,own-x2)
    return (x3-manda_phala(b,x3-u))%360
def planet_positions(jd_ut,variant="H"):
    m=mean_longitudes(jd_ut); t={"Sun":true_sun(m),"Moon":true_moon(m)}
    for b in ("Mars","Jupiter","Saturn"): t[b]=true_exterior(b,m)
    for b in ("Mercury","Venus"): t[b]=true_interior(b,m,variant)
    t["MoonNode"]=m["MoonNode"]; t["MoonApogee"]=m["MoonApogee"]
    return {"true":t,"mean":m,"ahargana":ahargana(jd_ut),"model":"Drgganita","variant":variant}

def mesa_ahargana(saka_year):
    """DG.ahargana (Part I ch.1 vv.7-8): ahargana at the mean Mesa sankranti of the given Saka year (current year number).
    Y = Saka + 3179 Kali years; days = Y*1461/4 * (1 + 1/42077) - 2. Checked: Saka 1924 (2002 CE) -> 1,863,913 (book's worked example 14 Jan 2002 = 1,863,823 plus 90 days)."""
    Y=saka_year+3179
    return Y*1461/4.0*(1+1/42077.0)-2.0
