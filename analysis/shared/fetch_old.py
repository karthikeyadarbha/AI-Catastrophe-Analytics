import requests,time,pandas as pd,io
rows=[]
for y0,y1 in ((1964,1969),(1970,1974),(1975,1979),(1980,1984),(1985,1989)):
    for t in range(6):
        r=requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query",params=dict(format="csv",starttime=f"{y0}-01-01",endtime=f"{y1}-12-31T23:59:59",minmagnitude=6.5,eventtype="earthquake",orderby="time-asc"),timeout=120)
        if r.status_code==200: rows.append(pd.read_csv(io.StringIO(r.text))); print(y0,y1,len(rows[-1])); break
        print(y0,r.status_code); time.sleep(5)
pd.concat(rows).to_csv("old_m65.csv",index=False)
