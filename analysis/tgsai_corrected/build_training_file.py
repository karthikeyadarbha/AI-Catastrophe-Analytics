"""Rebuild the megaquake training file with corrected columns.
usage: python build_training_file.py INPUT.csv OUTPUT.csv
Keeps the earthquake columns; replaces the sky columns with true sidereal positions, a sidereal node,
a computed Moon distance and the Moon-Sun phase angle. Rows before 1850 are dropped (planet elements valid ~1800-2050)."""
import sys,pandas as pd,numpy as np
import ephemeris as E, features as F
src,dst=sys.argv[1],sys.argv[2]
d=pd.read_csv(src); d["jd"]=F.to_jd(d["time"]); d=d[d.jd>=2396758.5].reset_index(drop=True)   # 1850-01-01
S=E.sidereal_positions(d.jd.values); out=d[["time","latitude","longitude","mag","place","depth","magType"]].copy()
out["Ayanamsa_Lahiri"]=E.ayan(d.jd.values)
for p,c in [("Moon","Sid_Moon_Lon"),("Sun","Sid_Sun_Lon"),("Mars","Sid_Mars_Lon"),("Jupiter","Sid_Jup_Lon"),("Saturn","Sid_Sat_Lon"),("Mercury","Sid_Mer_Lon"),("Venus","Sid_Ven_Lon"),("Node","Sid_Node_Lon")]: out[c]=np.round(S[p],4)
el=(S["Moon"]-S["Sun"])%360; out["Phase_Angle"]=np.round(np.minimum(el,360-el),4)
out["Moon_Dist_km"]=np.round(E.moon_distance_km(d.jd.values),0)
out.to_csv(dst,index=False); print("wrote",dst,len(out),"rows")
