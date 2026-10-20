import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
import sys, math, numpy as np, pandas as pd
sys.path.insert(0,".")
import eclipses as E
NAK=["Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra","Punarvasu","Pushya","Ashlesha","Magha","P.Phalguni","U.Phalguni","Hasta","Chitra","Svati","Vishakha","Anuradha","Jyeshtha","Mula","P.Ashadha","U.Ashadha","Shravana","Dhanishta","Shatabhisha","P.Bhadrapada","U.Bhadrapada","Revati"]
# region: (modern name, lat0,lat1,lon0,lon1, source grade)
R={
"Kalinga":("Coastal Odisha/N Andhra",15,22,81,87.5,"A"),
"Sala/Nishada/Kekaya":("Beas–Jhelum Punjab (Kekaya)",31,33.5,72,75.5,"A"),
"Suka/Kukkura":("W India/Gujarat–Rajasthan (Kukkura)",22,27,68,74,"B"),
"Gomati/Sindhu/Sauvira/Kuru-Panchala":("Sindh + Gomal valley",23,33,66,72,"A"),
"Kasi/Simhalaka":("Varanasi / Sri Lanka",5,26,79,84,"A"),
"Dandaka":("Deccan (Dandaka forest)",17,22,76,82,"B"),
"Kamboja/Surastra":("Kamboja NE Afghanistan; Surastra=Kathiawar/Gujarat",20,24.5,68,73,"A"),
"Magadha/Yavana":("S Bihar; Yavana=Bactria/Greeks",24,26.5,84,87.5,"A"),
"Dasarna":("E Malwa/Bhopal",22.5,25,77,80,"A"),
"Madra/Kurukshetra":("Sialkot; Delhi–Thanesar",28,33,73.5,78,"A"),
"Kashmira/Usinara":("Kashmir valley; Usinara Haridwar-Hazara hills",32.5,35.5,73,77.5,"A"),
"Asmaka":("Godavari upper, Telangana-Maharashtra",18,20,77,79.5,"B"),
"Kasi/Kosala":("Varanasi / Awadh",25,28,80,84,"A"),
"Daradas":("Dardistan (Gilgit–Chitral)",34.5,36.5,71.5,75,"A"),
"Malavaka/Yaudheya":("Malwa; Yaudheya=Sutlej–Bahawalpur",22,31,71,77,"A"),
"Pancanada/Suvira":("Punjab (five rivers); Sauvira=lower Indus",23,32,66,75,"A"),
"Arjunayana/Pundra/Sibi/Malava":("Alwar–Jaipur; N Bengal; Sibi (Shorkot)",24,31,72,89,"B"),
"Sattva/Avatamsaka":("unlocated",None,None,None,None,"C"),
"Saka":("Sakastan/Seistan (Helmand)",29,33,61,65,"A"),
"Kekaya/Panchala":("Beas–Jhelum; Rohilkhand",27,34,72,80,"A"),
"Vanga/Magadha/Kukkura":("Bengal; S Bihar",21,27,84,92,"A"),
"Asmaka/Kshudraka/Trigarta":("Telangana; Beas–Sutlej; Kangra–Jalandhar",18,33,75,79.5,"B"),
"Videha/Anarta/Sindhu/Sauvira":("Tirhut N Bihar; N Gujarat; Sindh",22,28,66,87,"A"),
"Horses/Bharani-south":("Southern countries",None,None,None,None,"C"),
}
nakmap={2:"Kalinga",3:None,4:"Sala/Nishada/Kekaya",5:"Suka/Kukkura",6:None,7:"Gomati/Sindhu/Sauvira/Kuru-Panchala",8:"Kasi/Simhalaka",9:"Dandaka",10:"Kamboja/Surastra",11:"Magadha/Yavana",12:"Dasarna",13:"Madra/Kurukshetra",14:"Kashmira/Usinara",15:"Asmaka",16:"Kasi/Kosala",17:"Daradas",18:"Malavaka/Yaudheya",19:"Pancanada/Suvira",20:"Arjunayana/Pundra/Sibi/Malava",21:"Sattva/Avatamsaka",22:"Saka",23:"Kekaya/Panchala",24:"Vanga/Magadha/Kukkura",25:"Asmaka/Kshudraka/Trigarta",26:"Videha/Anarta/Sindhu/Sauvira",0:"Horses/Bharani-south",1:"Kalinga"}
# regional list CSV
rows=[]
for i,n in enumerate(NAK):
    k=nakmap.get(i)
    if k is None: rows.append((n,"(no regional clause)","","","","","")); continue
    r=R[k]; rows.append((n,k,r[0],r[1],r[2],r[3],r[4])); 
pd.DataFrame(rows,columns=["Eclipse_Nakshatra","Ancient_regions (PT 5.4)","Modern_location","lat_min","lat_max","lon_min","lon_max"]).assign(Source_grade=[ (R[nakmap[i]][5] if nakmap.get(i) else "") for i in range(27)]).to_csv("results/eclipse_nakshatra_regions_modern.csv",index=False)
def sun_lon(jd):
    T=(jd-2451545.0)/36525
    L0=280.46646+36000.76983*T; M=math.radians(357.52911+35999.05029*T)
    C=(1.914602-0.004817*T)*math.sin(M)+0.019993*math.sin(2*M)+0.000289*math.sin(3*M)
    return (L0+C)%360
def ayan(jd): return 23.853+ (jd-2451545.0)/365.25*0.013972
def nak_of(e):
    jd=E.dt_to_jd(e["date"])
    lon=sun_lon(jd) if e["kind"]=="solar" else (sun_lon(jd)+180)%360
    return int(((lon-ayan(jd))%360)//(360/27))
ev=E.find(1996,2028)
for e in ev: e["nak"]=nak_of(e)
q=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv"); q["t"]=pd.to_datetime(q.UTC)
def inbox(k,lat,lon):
    r=R[k]
    return r[1] is not None and r[1]<=lat<=r[2] and r[3]<=lon<=r[4]
out=[]
for _,x in q.iterrows():
    hit=[];near=[]
    for e in ev:
        d=(pd.Timestamp(e["date"])-x.t).total_seconds()/86400
        if abs(d)<=30:
            k=nakmap.get(e["nak"]); near.append(f'{e["kind"]} {d:+.0f}d {NAK[e["nak"]]}')
            if k and inbox(k,x.Lat,x.Lon): hit.append(f'{NAK[e["nak"]]}->{k}')
    out.append((x.Event,"; ".join(near),"; ".join(hit)))
o=pd.DataFrame(out,columns=["Event","eclipses_within_30d","region_match"]); print(o.to_string())
o.to_csv("results/eclipse_region_test.csv",index=False)
# Monte Carlo: in-scope quakes random date same location
scope=q[(q.Lon>55)&(q.Lon<90)&(q.Lat>20)&(q.Lat<40)]
print(list(scope.Event))
rng=np.random.default_rng(2); jd0=E.dt_to_jd(pd.Timestamp("1999-09-01")); 
edays=np.array([E.dt_to_jd(e["date"]) for e in ev]); enak=np.array([e["nak"] for e in ev])
def hits(lat,lon,jd):
    m=np.abs(edays-jd)<=30
    return any(nakmap.get(n) and inbox(nakmap[n],lat,lon) for n in enak[m])
obs=sum(hits(r.Lat,r.Lon,E.dt_to_jd(r.t.to_pydatetime())) for _,r in scope.iterrows())
sims=[sum(hits(r.Lat,r.Lon,jd0+rng.integers(0,9000)) for _,r in scope.iterrows()) for _ in range(5000)]
print("obs",obs,"of",len(scope),"MC mean",np.mean(sims),"p>=",np.mean(np.array(sims)>=obs))
