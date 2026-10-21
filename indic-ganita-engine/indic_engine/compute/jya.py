"""Rsine (jya) and arc (capa) under five schemes. Arcs and sines in minutes of arc, radius R = trijya.
Table schemes follow Tantrasangraha 2.2 (interpolation between 24 tabular jyas at 225' steps, one rule for jya, the reverse rule for capa).
Table values: Tantrasangraha Table 2.1 (columns 'AR/SS', 'TS', 'Madhava'); 'series' is the Madhava-Newton sine series (Ganita-yukti-bhasa, Tantrasangraha 2.3-2.4); 'exact' is R sin."""
import math
def _hms(*t): return t[0]+t[1]/60+t[2]/3600
ARYABHATA=[225,449,671,890,1105,1315,1520,1719,1910,2093,2267,2431,2585,2728,2859,2978,3084,3177,3256,3321,3372,3409,3431,3438]
TANTRASANGRAHA=[_hms(*x) for x in [(224,50,21),(448,42,58),(670,40,16),(887,45,17),(1105,1,41),(1315,34,11),(1520,28,41),(1718,52,32),(1909,54,46),(2092,46,19),(2266,40,10),(2430,51,40),(2585,38,37),(2727,21,31),(2858,23,42),(2977,11,30),(3083,14,23),(3176,5,7),(3255,19,50),(3320,38,11),(3371,43,24),(3408,22,20),(3430,25,35),(3437,47,29)]]
MADHAVA=[_hms(*x) for x in [(224,50,22),(448,42,58),(670,40,16),(889,45,15),(1105,1,39),(1315,34,7),(1520,28,35),(1718,52,24),(1909,54,35),(2092,46,3),(2266,39,50),(2430,51,15),(2584,38,6),(2727,20,52),(2858,22,55),(2977,10,34),(3083,13,17),(3176,3,50),(3255,18,22),(3320,36,30),(3371,41,29),(3408,20,11),(3430,23,11),(3437,44,48)]]
SCHEMES={"aryabhata":(ARYABHATA,3438.0),"tantrasangraha":(TANTRASANGRAHA,TANTRASANGRAHA[-1]),"madhava":(MADHAVA,MADHAVA[-1])}
R_EXACT=21600/(2*math.pi)
def trijya(scheme):
    return SCHEMES[scheme][1] if scheme in SCHEMES else R_EXACT
def _jya_first_quadrant(S,scheme):
    if scheme=="exact": return R_EXACT*math.sin(S/R_EXACT)
    if scheme=="series":
        x=S/R_EXACT; t=x; s=x; n=1
        while abs(t)>1e-16 and n<40: t*=-x*x/((2*n)*(2*n+1)); s+=t; n+=1   # Madhava-Newton series: sum (-1)^n x^(2n+1)/(2n+1)!
        return R_EXACT*s
    tab,_=SCHEMES[scheme]; i=int(S//225); r=S-225*i
    if i>=24: return tab[-1]
    lo=0.0 if i==0 else tab[i-1]; return lo+r*(tab[i]-lo)/225.0     # Tantrasangraha 2.2, rule of three
def jya(arc,scheme="madhava"):
    """signed Rsine of any arc (minutes)"""
    a=arc%21600.0; s=1.0
    if a>10800: a-=10800; s=-1.0
    if a>5400: a=10800-a
    return s*_jya_first_quadrant(a,scheme)
def kotijya(arc,scheme="madhava"): return jya(arc+5400.0,scheme)
def capa(j,scheme="madhava"):
    """arc (minutes, in [-5400,5400]) from a signed Rsine, by the reverse of the rule used for jya"""
    s=1.0 if j>=0 else -1.0; j=abs(j); lo,hi=0.0,5400.0
    if j>=_jya_first_quadrant(5400.0,scheme): return s*5400.0
    for _ in range(80):
        m=(lo+hi)/2
        if _jya_first_quadrant(m,scheme)<j: lo=m
        else: hi=m
    return s*(lo+hi)/2
