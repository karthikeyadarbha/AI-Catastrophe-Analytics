import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import numpy as np, pandas as pd
from scipy import stats
# sidereal Lahiri longitudes from astral-ephemeris: Sun,Moon,Mars,Mercury,Jupiter,Venus,Saturn
D={"Sumatra 2004":[250.6705,64.2658,216.3301,228.837,172.7517,227.7225,91.4605],
"Haiti 2010":[268.5882,242.4706,112.0437,252.1358,304.8588,268.8349,160.6501],
"Sichuan 2008":[27.9323,119.3004,97.3249,49.459,268.3797,20.4303,127.778],
"Kashmir 2005":[171.0345,227.4059,29.1437,185.2029,182.1658,216.1953,105.5502],
"Turkey-Syria 2023":[292.7902,115.8888,47.2864,268.7013,342.8479,318.1846,302.2768],
"Bam 2003":[249.9498,286.766,341.6924,252.2566,144.8672,282.0227,76.3301],
"Bhuj 2001":[282.3649,299.5285,205.4577,300.4924,37.3208,329.1693,30.1929],
"Tohoku 2011":[326.2878,36.1125,318.7627,338.7939,346.1128,286.8025,171.6168],
"Izmit 1999":[119.8231,186.8459,206.0936,101.3137,11.0298,125.329,23.1817],
"Nepal 2015":[10.6636,92.3231,23.8958,26.5006,108.9429,51.6629,219.5041],
"Venezuela 2026":[69.1769,193.5291,32.9864,91.1412,94.6408,109.0322,349.6676],
"Java 2006":[41.5969,38.1155,91.3471,51.3497,197.3529,3.0558,102.6894],
"Sulawesi 2018":[161.1159,20.5968,280.8485,166.9381,207.4825,195.7206,248.8192],
"Myanmar 2025":[343.6122,326.9364,88.2135,337.2186,51.2035,335.268,329.8291],
"Morocco 2023":[141.8146,74.5707,163.7916,137.1505,21.3645,108.4707,308.7096]}
q=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv").set_index("Event")
rows=[]
for ev,L in D.items():
    L=np.array(L); sun,moon=L[0],L[1]
    el=(moon-sun)%360; tithi=int(el//12)+1            # 1..30
    t15=(tithi-1)%15+1                                  # PT names cycle of 15
    tname=["Nanda","Bhadra","Jaya","Rikta","Purna"][(t15-1)%5]
    k=int(el//6)+1                                      # half-tithi 1..60
    if k==1: kar="Kimstughna"
    elif k==58: kar="Shakuni"
    elif k==59: kar="Chatushpada"
    elif k==60: kar="Naga"
    else: kar=["Bava","Balava","Kaulava","Taitila","Gara","Vanija","Vishti"][(k-2)%7]
    nak=int(moon//(360/27))
    syz=(1+np.cos(np.radians(2*el)))/2
    pl=L[2:]  # Mars,Mer,Jup,Ven,Sat
    names=["Mars","Mercury","Jupiter","Venus","Saturn"]
    pairs=[(abs((pl[i]-pl[j]+180)%360-180),names[i]+"-"+names[j]) for i in range(5) for j in range(i+1,5)]
    pm=min(pairs)
    sep=[abs((moon-x+180)%360-180) for x in L[2:]]+[abs((moon-sun+180)%360-180)]
    naks=(L[2:]//(360/27)).astype(int)
    clus=np.bincount(naks).max()
    rows.append(dict(Event=ev,Moon_Sun_elong=round(el,1),Tithi30=tithi,PT_tithi=tname,Karana=kar,Moon_nak_idx=nak,Syzygy_recalc=round(syz,3),
      Min_planet_pair_deg=round(pm[0],1),Closest_pair=pm[1],Moon_min_sep_planet_deg=round(min(sep),1),Same_nak_cluster_5planets=int(clus)))
df=pd.DataFrame(rows).set_index("Event").join(q[["Deaths","Mag","Lunar_Syzygy","LocalSolarHour","Moon_Nakshatra"]])
print(df.drop(columns=["Moon_nak_idx"]).to_string())
print("syzygy corr repo vs recalculated:",np.corrcoef(df.Syzygy_recalc,df.Lunar_Syzygy)[0,1].round(3))
rng=np.random.default_rng(7); N=200000
def pmc(obs,p,n=15):  # upper-tail binomial
    return stats.binom.sf(obs-1,n,p)
res=[]
# tithi classes
for nm in ["Nanda","Bhadra","Jaya","Rikta","Purna"]:
    o=(df.PT_tithi==nm).sum(); res.append((f"tithi {nm}",o,15*.2,pmc(o,.2)))
o=(df.Karana=="Vishti").sum(); res.append(("karana Vishti",o,15*8/60,pmc(o,8/60)))
fixed=df.Karana.isin(["Shakuni","Chatushpada","Naga","Kimstughna"]).sum(); res.append(("fixed karanas",fixed,15*4/60,pmc(fixed,4/60)))
# sandhya +/-48 min
h=df.LocalSolarHour; sand=((abs(h-6)<=0.8)|(abs(h-18)<=0.8)).sum(); res.append(("within 1 muhurta of sunrise/sunset (solar-time approx)",sand,15*3.2/24,pmc(sand,3.2/24)))
# nakshatra groups
G={"Dhruva":[3,11,20,25],"Mridu":[4,13,16,26],"Ugra":[9,1,10,19,24],"Tikshna(Ardra,Ashlesha,Jyeshtha,Mula)":[5,8,17,18],"Kshipra":[7,0,12],"Chara":[6,14,21,22,23],"Common(Krittika,Vishakha)":[2,15]}
# 0-based nak idx: Ashwini0 Bharani1 Krittika2 Rohini3 Mrig4 Ardra5 Punar6 Pushya7 Ashl8 Magha9 PPh10 UPh11 Hasta12 Chitra13 Svati14 Vish15 Anur16 Jye17 Mula18 PAs19 UAs20 Shr21 Dha22 Sat23 PBh24 UBh25 Rev26
G={"Dhruva":[3,11,20,25],"Mridu":[4,13,16,26],"Ugra":[9,1,10,19,24],"Tikshna(Ardra,Ashlesha,Jyeshtha,Mula)":[5,8,17,18],"Kshipra":[7,0,12],"Chara":[6,14,21,22,23],"Common(Krittika,Vishakha)":[2,15]}
for g,m in G.items():
    o=df.Moon_nak_idx.isin(m).sum(); res.append((f"Moon nak group {g}",o,15*len(m)/27,pmc(o,len(m)/27)))
# Moon-planet within 5 deg (upasarga) MC: moon uniform
obs=(df.Moon_min_sep_planet_deg<=5).sum()
Ls=np.array(list(D.values()))
sims=np.zeros(N)
for ev_i in range(15):
    mm=rng.uniform(0,360,N); others=Ls[ev_i,[0,2,3,4,5,6]]
    sep=np.abs((mm[:,None]-others[None,:]+180)%360-180).min(1)
    sims+= (sep<=5)
res.append(("Moon within 5deg of a planet/Sun (upasarga)",obs,sims.mean(),(sims>=obs).mean()))
# planet war <=1 and <=3 deg
for thr in (1,3):
    obs=(df.Min_planet_pair_deg<=thr).sum(); s=np.zeros(N)
    for ev_i in range(15):
        x=rng.uniform(0,360,(N,5)); d=np.abs((x[:,:,None]-x[:,None,:]+180)%360-180)
        iu=np.triu_indices(5,1); s+=(d[:,iu[0],iu[1]].min(1)<=thr)
    res.append((f"planetary war pair <= {thr}deg (5 tara-grahas)",obs,s.mean(),(s>=obs).mean()))
# >=3 of 5 planets same nakshatra
obs=(df.Same_nak_cluster_5planets>=3).sum(); s=np.zeros(N)
for ev_i in range(15):
    x=rng.integers(0,27,(N,5)); c=np.zeros(N)
    for v in range(27): c=np.maximum(c,(x==v).sum(1))
    s+=(c>=3)
res.append(("3+ of 5 planets in one nakshatra (Srngataka/Vyatha-like)",obs,s.mean(),(s>=obs).mean()))
r=pd.DataFrame(res,columns=["test","observed_of_15","expected","p_upper"]).round(3)
r["p_bonferroni"]=(r.p_upper*len(r)).clip(upper=1).round(3)
print(r.to_string())
# composite score vs deaths
df["score"]=((df.PT_tithi=="Rikta")*1+(df.Karana=="Vishti")*1+df.Moon_nak_idx.isin([5,8,17,18])*1+(df.Min_planet_pair_deg<=3)*1+(df.Moon_min_sep_planet_deg<=5)*1)
print(df[["score","Deaths","Mag"]].T.to_string())
print("Spearman score~deaths",stats.spearmanr(df.score,df.Deaths))
print("Spearman score~mag",stats.spearmanr(df.score,df.Mag))
df.drop(columns=["Moon_nak_idx"]).to_csv("results/ch14_16_features_15_quakes.csv")
r.to_csv("results/ch14_16_tests.csv",index=False)
