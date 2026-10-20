import requests,time,pandas as pd,io,calendar
from concurrent.futures import ThreadPoolExecutor
def wins_A():
    w=[]
    for y in range(1964,1990):
        for m in range(1,13,2): w.append((f"{y}-{m:02d}-01",f"{y}-{m+1:02d}-{calendar.monthrange(y,m+1)[1]}T23:59:59",5.5,6.5))
    return w
def wins_B(): return [(f"{y}-01-01",f"{y+4}-12-31T23:59:59",7.0,None) for y in range(1900,1964,5)]
def get(w):
    p=dict(format="csv",starttime=w[0],endtime=w[1],minmagnitude=w[2],eventtype="earthquake",orderby="time-asc")
    if w[3]: p["maxmagnitude"]=w[3]
    for t in range(10):
        try:
            r=requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query",params=p,timeout=90)
            if r.status_code==200: return pd.read_csv(io.StringIO(r.text))
        except Exception: pass
        time.sleep(3+t)
    print("FAILED",w,flush=True); return None
for name,W in (("histA_m55_65.csv",wins_A()),("histB_m7.csv",wins_B())):
    with ThreadPoolExecutor(4) as ex: out=list(ex.map(get,W))
    g=[d for d in out if d is not None]; pd.concat(g).to_csv(name,index=False); print(name,sum(len(d) for d in g),"failed",len(out)-len(g),flush=True)
