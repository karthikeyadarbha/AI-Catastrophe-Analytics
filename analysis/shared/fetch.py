import pandas as pd, subprocess, time, io, datetime as dt, sys
ev=pd.read_csv("results/earthquakes_parashara_tantra_mandalas.csv")
out=[]
for r in ev.itertuples():
    t=dt.datetime.strptime(r.UTC,"%Y-%m-%d %H:%M:%S")
    a=(t-dt.timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S"); b=(t+dt.timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%S")
    url=f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={a}&endtime={b}&minmagnitude=4.5&latitude={r.Lat}&longitude={r.Lon}&maxradiuskm=300&orderby=time-asc&limit=20000"
    for k in range(6):
        p=subprocess.run(["curl","-s","-m","90","-w","\n%{http_code}",url],capture_output=True,text=True)
        body,code=p.stdout.rsplit("\n",1)
        if code=="200": break
        time.sleep(3)
    if code!="200": print("FAIL",r.Event,code); continue
    d=pd.read_csv(io.StringIO(body)); d.insert(0,"MainShock",r.Event); d.insert(1,"MainShock_UTC",r.UTC)
    cnt=subprocess.run(["curl","-s","-m","60",url.replace("/query?format=csv","/count?format=text").split("&orderby")[0]],capture_output=True,text=True).stdout.strip()
    print(r.Event,len(d),"count endpoint:",cnt,flush=True); out.append(d)
pd.concat(out).to_csv("usgs_catalogue_15_events.csv",index=False)
