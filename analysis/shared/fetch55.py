import requests,time,pandas as pd,io,sys
rows=[]
for y in range(1990,2027):
    for half in ((f"{y}-01-01",f"{y}-06-30T23:59:59"),(f"{y}-07-01",f"{y}-12-31T23:59:59")):
        for t in range(8):
            try:
                r=requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query",params=dict(format="csv",starttime=half[0],endtime=half[1],minmagnitude=5.5,maxmagnitude=6.5,eventtype="earthquake",orderby="time-asc"),timeout=120)
                if r.status_code==200: df=pd.read_csv(io.StringIO(r.text)); rows.append(df); print(half[0],len(df),flush=True); break
                print(half[0],r.status_code,flush=True)
            except Exception as e: print(half[0],"err",e,flush=True)
            time.sleep(4)
pd.concat(rows).to_csv("m55_65.csv",index=False); print("done")
