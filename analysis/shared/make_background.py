"""Write bg_sid.csv: the 6-hourly background grid (datetime only), 1989-12-15 to 2026-11-01.
The Modern M5.5+ scripts need only this column. The Modern M6.5+ scripts (big_test, extra*, ayan, stage1_*)
also need tool-derived sidereal/tropical features for the grid and the events (bg_sid.csv, bg_trop.csv,
quakes_m65_sid.csv, quakes_m65_trop.csv) that came from the astral-ephemeris tool; they are not included (about 7.5 MB each)."""
import pandas as pd
g=pd.DataFrame({"datetime":pd.date_range("1989-12-15","2026-11-01",freq="6h").strftime("%Y-%m-%dT%H:%M:%SZ")})
g.to_csv("bg_sid.csv",index=False); print(len(g),"rows")
