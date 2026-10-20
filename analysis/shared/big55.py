import sys,numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,"."); import myeph
sys.path.insert(0,"../shared"); import eclipses as E
W=360/27
Q=pd.read_csv("quakes_m55_input.csv"); Q["t"]=pd.to_datetime(Q.datetime).dt.tz_localize(None)
bg=pd.read_csv("bg_sid.csv")[["datetime"]]; bg["t"]=pd.to_datetime(bg.datetime).dt.tz_localize(None)
S=pd.concat([Q[["id"]],myeph.enrich(Q)],axis=1); B=myeph.enrich(bg)
jdq=2440587.5+(Q.t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400; jdb=2440587.5+(bg.t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
S["ay"]=myeph.ayan(jdq); B["ay"]=myeph.ayan(jdb)
S.to_csv("m55_features_sid.csv",index=False); B.to_csv("bg_features_myeph.csv",index=False)
print("events",len(Q),"bg",len(bg))
