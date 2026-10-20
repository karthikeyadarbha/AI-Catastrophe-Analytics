import numpy as np,pandas as pd
a=pd.read_csv('m55_65.csv'); b=pd.read_csv('big.csv')
q=pd.concat([a,b]).drop_duplicates('id'); q=q[(q.type=='earthquake')&(q.mag>=5.5)].copy()
q['t']=pd.to_datetime(q.time).dt.tz_localize(None); q=q.sort_values('mag',ascending=False).reset_index(drop=True)
print('raw',len(q))
keep=np.ones(len(q),bool); lat=np.radians(q.latitude.values); lon=np.radians(q.longitude.values); tt=q.t.values.astype('datetime64[s]').astype(np.int64)/86400.0; mg=q.mag.values
sl,cl=np.sin(lat),np.cos(lat)
for i in range(len(q)):
    if not keep[i]: continue
    M=mg[i]; T=10**(0.032*M+2.7389) if M>=6.5 else 10**(0.5409*M-0.547); R=10**(0.1238*M+0.983)
    m=(np.abs(tt-tt[i])<=T)&(mg<=M); m[i]=False; m&=keep
    idx=np.where(m)[0]
    if len(idx)==0: continue
    d=6371*np.arccos(np.clip(sl[idx]*sl[i]+cl[idx]*cl[i]*np.cos(lon[idx]-lon[i]),-1,1)); keep[idx[d<=R]]=False
Q=q[keep].sort_values('t').reset_index(drop=True); print('declustered',len(Q), (Q.mag>=6.5).sum())
Q['datetime']=Q.t.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
Q[['id','datetime','latitude','longitude','mag','depth']].rename(columns={'latitude':'lat','longitude':'lon'}).to_csv('quakes_m55_input.csv',index=False)
print(Q.mag.describe().round(2).to_dict())
