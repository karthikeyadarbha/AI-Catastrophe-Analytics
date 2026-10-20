import pandas as pd, subprocess, time, io, datetime as dt, concurrent.futures as cf
Q=pd.read_csv("results/declustered_events_M65.csv"); Q=Q[Q.mag>=7.0].reset_index(drop=True)
print("M7+ events",len(Q),flush=True)
def get(i):
    r=Q.iloc[i]; t=pd.to_datetime(r.time).tz_localize(None)
    a=t.strftime("%Y-%m-%dT%H:%M:%S"); b=(t+pd.Timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%S")
    rad=max(100,min(300,int(60*10**(0.5*(r.mag-7)))+100))
    url=f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={a}&endtime={b}&minmagnitude=5.0&latitude={r.latitude}&longitude={r.longitude}&maxradiuskm={rad}&orderby=time-asc&limit=20000"
    for k in range(8):
        p=subprocess.run(["curl","-s","-m","90","-w","\n%{http_code}",url],capture_output=True,text=True)
        if p.stdout.count("\n")>=0 and p.stdout.rsplit("\n",1)[-1]=="200":
            d=pd.read_csv(io.StringIO(p.stdout.rsplit("\n",1)[0])); d.insert(0,"MainShock",r.id); d.insert(1,"MS_time",r.time); d.insert(2,"MS_mag",r.mag); return d
        time.sleep(2+k)
    return None
out=[]
with cf.ThreadPoolExecutor(4) as ex:
    for i,d in enumerate(ex.map(get,range(len(Q)))):
        if d is not None: out.append(d)
        if i%25==0: print(i,len(out),flush=True)
pd.concat(out).to_csv("aftershocks_M7plus.csv",index=False); print("done",len(out),flush=True)
