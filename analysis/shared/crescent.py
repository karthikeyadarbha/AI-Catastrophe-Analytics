import sys,numpy as np,pandas as pd
sys.path.insert(0,"../shared"); import myeph as M; from moonlat import moon_lat
from hist import grid
R=np.radians; D=np.degrees
def jd_of(t):
    t=pd.Series(pd.to_datetime(t)).reset_index(drop=True); return 2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
def eq_vec(lon,lat,jd):
    T=(jd-2451545.0)/36525; eps=R(23.4392911-0.0130042*T); l=R(lon); b=R(lat)
    x=np.cos(b)*np.cos(l); y=np.cos(b)*np.sin(l)*np.cos(eps)-np.sin(b)*np.sin(eps); z=np.cos(b)*np.sin(l)*np.sin(eps)+np.sin(b)*np.cos(eps); return np.stack([x,y,z],-1)
def local(v,jd,lat,lon):
    th=R((280.46061837+360.98564736629*(jd-2451545.0)+lon)%360); ph=R(lat)
    e=-np.sin(th)*v[...,0]+np.cos(th)*v[...,1]; n=-np.sin(ph)*np.cos(th)*v[...,0]-np.sin(ph)*np.sin(th)*v[...,1]+np.cos(ph)*v[...,2]; u=np.cos(ph)*np.cos(th)*v[...,0]+np.cos(ph)*np.sin(th)*v[...,1]+np.sin(ph)*v[...,2]
    return np.stack([e,n,u],-1)
def sunv(jd): return eq_vec(M.sun_trop(jd),0*jd,jd)
def moonv(jd): return eq_vec(M.moon_trop(jd),moon_lat(jd),jd)
def elong(jd): return (M.moon_trop(jd)-M.sun_trop(jd))%360
def conj_before(jd):
    t=jd-elong(jd)/12.1907
    for _ in range(3): t=t-(((elong(t)+180)%360)-180)/12.1907
    return np.where(t>jd,t-29.53,t)   # safety
def sight(tc,lat,lon,ndays=4):
    N=len(tc); found=np.zeros(N,bool); tilt=np.full(N,np.nan); az=np.full(N,np.nan); steps=np.arange(0,9.01,1/6)  # hours after local solar 15h
    for k in range(ndays):
        todo=~found
        if not todo.any(): break
        idx=np.where(todo)[0]; base=np.floor(tc[idx]+0.5-0.0)+k   # JD of local-noon-ish day (UT noon = jd .0)
        noon=base-lon[idx]/360.0           # local solar noon (UT days, JD integer = UT noon)
        jd=noon[:,None]+(3+steps[None,:])/24.0; sun=local(sunv(jd),jd,lat[idx][:,None],lon[idx][:,None]); alt=D(np.arcsin(np.clip(sun[...,2],-1,1)))
        ok=alt<=-3; has=ok.any(1); j=np.argmax(ok,1); tj=jd[np.arange(len(idx)),j]
        ms=local(moonv(tj),tj,lat[idx],lon[idx]); ss=local(sunv(tj),tj,lat[idx],lon[idx]); malt=D(np.arcsin(np.clip(ms[:,2],-1,1)))
        el=elong(tj); good=has&(malt>=3)&(el>=8)&(el<=60)
        zen=np.array([0,0,1.0]); st=ss-(ss*ms).sum(1,keepdims=True)*ms; zt=zen-(ms[:,2:3])*ms
        cos=(st*(-zt)).sum(1)/np.maximum(np.linalg.norm(st,axis=1)*np.linalg.norm(zt,axis=1),1e-12); tl=D(np.arccos(np.clip(cos,-1,1))); a=D(np.arctan2(ms[:,0],ms[:,1]))%360
        sel=idx[good]; found[sel]=True; tilt[sel]=tl[good]; az[sel]=a[good]
    return found,tilt,az
def flags(found,tilt,az):
    lev=found&(tilt<=20); yoke=lev&(np.abs(az-270)<=30)|lev&(np.abs(az-90)<=30); return yoke,lev
if __name__=="__main__":
    # sanity: sight at Ujjain over one year of months; print share of months flagged
    k=np.arange(0,150); tc=2460000+29.5306*k; tc=conj_before(tc+1.0)
    f,t,a=sight(tc,np.full(len(tc),23.18),np.full(len(tc),75.78)); y,l=flags(f,t,a); print("months",len(tc),"sighted",f.mean().round(3),"yoke",y.mean().round(3),"level",l.mean().round(3)); print(pd.Series(t[f]).describe().round(1).to_dict()); print(pd.Series(a[f]).describe().round(1).to_dict())
