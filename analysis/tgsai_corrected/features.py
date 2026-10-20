"""Corrected feature builder for TG-SAI (see ../../docs/audit/README_AUDIT.md for what was wrong before)."""
import numpy as np
import ephemeris as E
r=np.radians
def _sc(x): return [np.sin(r(x)),np.cos(r(x))]
def vedic_features(sid):
    nk=(sid["Moon"]//(360/27)).astype(int); oh=np.eye(27)[nk]; el=(sid["Moon"]-sid["Sun"])%360
    return np.c_[oh[:,[14,22,0,11,12,13]].sum(1),oh[:,[2,15,24,1,19,9,7,10]].sum(1),oh[:,[17,20,21,3,4,6,16]].sum(1),
                 oh[:,[23,25,26,5,8,18]].sum(1),oh[:,[5,8,17,18]].sum(1),np.floor(el/12)]
def feature_matrix(jd,vedic=False,node_frame="sidereal"):
    """Physical features from TRUE planet positions, a sidereal node, a computed Moon distance (perigee).
    node_frame='tropical' reproduces the original file's frame mismatch, for comparison only."""
    sid=E.sidereal_positions(jd); A=E.all_trop(jd)
    dec=np.degrees(np.arcsin(np.sin(r(23.439))*np.sin(r(A["Moon"]))))
    el=(sid["Moon"]-sid["Sun"])%360; phase=np.minimum(el,360-el)
    cols=[np.abs(dec),np.abs(dec)/10,phase,E.moon_distance_km(jd)]
    for p in ["Moon","Sun","Mars","Jupiter","Saturn"]: cols+=_sc(sid[p])
    cols+=_sc(sid["Node"] if node_frame=="sidereal" else E.node_trop(jd))
    X=np.c_[tuple(cols)]
    return np.c_[X,vedic_features(sid)] if vedic else X
def to_jd(series):
    import pandas as pd
    t=pd.to_datetime(series,format="ISO8601",utc=True).dt.tz_localize(None)
    return 2440587.5+(t-pd.Timestamp("1970-01-01")).dt.total_seconds().values/86400
