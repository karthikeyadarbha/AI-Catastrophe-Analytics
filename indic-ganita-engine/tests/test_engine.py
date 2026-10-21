"""Tests: the engine must reproduce the worked examples printed in Ramasubramanian & Sriram, Tantrasangraha (2011), and internal consistency."""
import math, sys, os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))
from indic_engine.compute import params as P
from indic_engine.compute.jya import jya, capa, ARYABHATA, R_EXACT
from indic_engine.compute.positions import ahargana, mean_longitudes
def test_ahargana_example():            # TS 1.12 example: 14 Jan 2002 -> Ahargana 1863823 (JD 2452288.5 at 0h)
    jd0=2452288.5
    assert abs((jd0-P.KALI_EPOCH_JD_MIDNIGHT)-1863823)<1e-9
def test_moon_revolutions_example():    # TS (1.12): 1863823*57753320/1577917500 = 68217.7402447
    n=1863823*P.REV["TS"]["Moon"]/P.CIVIL_DAYS_MAHAYUGA; assert abs(n-68217.7402447)<1e-6
    f=n%1; assert int(f*12)==8 and int((f*12%1)*30)==26 and int(((f*12%1)*30%1)*60)==29   # 8 signs 26 deg 29'
def test_tabular_jya_interpolation():   # TS 2.2 worked example: arc 1947' with Madhava J8,J9 -> 1843'41"
    from indic_engine.compute.jya import MADHAVA
    j=MADHAVA[7]+147*(MADHAVA[8]-MADHAVA[7])/225
    assert abs(j-(1843+41/60+2/3600))<0.02
def test_aryabhata_table_second_differences():
    d=[ARYABHATA[0]]+[ARYABHATA[i]-ARYABHATA[i-1] for i in range(1,24)]
    assert d[:5]==[225,224,222,219,215] and d[-1]==7
def test_series_matches_exact():
    for deg in (1,17,45,89,90):
        assert abs(jya(deg*60,"series")-R_EXACT*math.sin(math.radians(deg)))<1e-6
def test_capa_inverse():
    for sch in ("aryabhata","tantrasangraha","madhava","series","exact"):
        for a in (100.0,2000.0,5000.0): assert abs(capa(jya(a,sch),sch)-a)<1e-3
def test_node_moves_backward():
    a=mean_longitudes(2452288.5)["MoonNode"]; b=mean_longitudes(2452288.5+10)["MoonNode"]; assert ((a-b)%360)<2 and ((a-b)%360)>0.1
if __name__=="__main__":
    for n,f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok",n)
