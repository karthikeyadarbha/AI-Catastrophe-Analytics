"""Frozen parameters. Every constant carries the rule ID of the verse (or table) it comes from (see rules/data/*.json)."""
from .jya import _hms
CIVIL_DAYS_MAHAYUGA=1577917500            # TS 1.10 v.19
SUN_REV=4320000                           # TS 1.9 v.15
KALI_EPOCH_JD_MIDNIGHT=588465.5           # Ahargana of TS's worked example (14 Jan 2002 = 1863823) fixes this: 2452288.5-1863823
UJJAIN_LON=75.7684                        # external constant: meridian of Ujjayini (TS 1.12 note); not given in the verses
SUNRISE_LOCAL_H=6.0                       # mean sunrise = 06:00 local mean time at Ujjayini (TS 1.12 'mean sunrise')
# revolutions in a mahayuga: TS 1.9 vv.15-18 (Nilakantha); AB = Aryabhatiya Gitikapada 3-4 values as quoted in TS (footnote 36) and verified against the scan
REV={"TS":{"Sun":4320000,"Moon":57753320,"Mars":2296864,"Mercury":17937048,"Jupiter":364180,"Venus":7022268,"Saturn":146612,"MoonApogee":488122,"MoonNode":232300},
     "AB":{"Sun":4320000,"Moon":57753336,"Mars":2296824,"Mercury":17937020,"Jupiter":364224,"Venus":7022388,"Saturn":146564,"MoonApogee":488219,"MoonNode":232226}}
# dhruvas at the Kali epoch: TS 1.15 vv.35-37, Table 1.7 (degrees). AB assumes all mean planets at 0 deg at the Kali epoch (TS 1.15 commentary)
DHRUVA={"TS":{"Sun":0.0,"Moon":_hms(4,45,46),"Mars":330+17+47/60,"Mercury":-36/60,"Jupiter":_hms(12,10,0),"Venus":30+_hms(6,13,0),"Saturn":330+_hms(17,20,0),"MoonApogee":90+_hms(29,17,5)+0,"MoonNode":180+_hms(22,20,0)+0},
        "AB":{**{k:0.0 for k in ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]},"MoonApogee":90.0,"MoonNode":180.0}}  # equal-pada assumption: Kali start = 3/4 yuga
DHRUVA["TS"]["Mars"]=11*30+_hms(17,47,0); DHRUVA["TS"]["Saturn"]=11*30+_hms(17,20,0); DHRUVA["TS"]["Venus"]=1*30+_hms(6,13,0); DHRUVA["TS"]["MoonApogee"]=3*30+_hms(29,17,5); DHRUVA["TS"]["MoonNode"]=6*30+_hms(22,20,0)
DHRUVA["TS"]["Moon"]=_hms(4,45,46)
MANDOCCA={"Sun":78.0,"Mars":127.0,"Mercury":220.0,"Jupiter":172.0,"Venus":80.0,"Saturn":240.0}   # TS 1.17 v.40, Table 1.8 (fixed)
PLANETS=["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
