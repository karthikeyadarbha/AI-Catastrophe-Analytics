import sys as _s, os as _o
_d=_o.path.dirname(_o.path.abspath(__file__))
while not _o.path.isdir(_o.path.join(_d,"shared")): _d=_o.path.dirname(_d)
_s.path.insert(0,_o.path.join(_d,"shared"))   # shared helpers (myeph, eclipses, hist, ...)
exec(open(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)),"adsagara.py")).read().split("rows=[]\nQ=pd.read_csv")[0])
Q=pd.read_csv("quakes_m55_input.csv"); bg=pd.read_csv("bg_sid.csv")
t=pd.to_datetime(Q.datetime).dt.tz_localize(None); tb=pd.to_datetime(bg.datetime).dt.tz_localize(None).reset_index(drop=True)
rng=np.random.default_rng(11); n=len(t); ix=rng.integers(0,n,len(tb)); lat,lon=Q.lat.values,Q.lon.values
he,sr,ss,ok=solar(t,lat,lon); hb,srb,ssb,okb=solar(tb,lat[ix],lon[ix])
for off in [-6,-4,-2,0,2,4,6]:
    f=lambda h,s,o:(abs(((h-s-off+12)%24)-12)<=0.8)&o
    e=f(he,sr,ok); b=f(hb,srb,okb); print("sunrise+%3d h window: %.3f"%(off,e[ok].mean()/b[okb].mean()))
# hour-of-day profile of events vs background in local solar hour
he2=np.floor(he).astype(int); hb2=np.floor(hb).astype(int)
print(np.round(np.bincount(he2,minlength=24)/len(he)/(np.bincount(hb2,minlength=24)/len(hb)),2))
