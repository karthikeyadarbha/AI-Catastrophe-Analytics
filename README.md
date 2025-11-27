```markdown
# Compute Sidereal Astrological Columns for CSV

This repository contains a Python script `compute_sidereal_planets.py` that reads a CSV with earthquake event rows (must include `time`, `latitude`, `longitude`) and appends sidereal astrological columns for Sun, Moon, Mars, Saturn, Venus, and both versions of the lunar nodes (Rahu_mean, Ketu_mean, Rahu_true, Ketu_true).

Default behavior (can be changed in script):
- Ayanamsha: Lahiri (sidereal) — explicitly set
- Nodes: both mean node and true node are computed (Rahu_mean / Ketu_mean and Rahu_true / Ketu_true)
- Retrograde: determined from ecliptic longitude speed < 0 (deg/day)
- Combustion: angular separation from Sun < 8.5°
- Rising: topocentric altitude > 0° (above horizon)
- Zodiac indexing: 1..12 (1 = Aries)

Dependencies:
- Python 3.8+
- pip packages:
  - pyswisseph  (pip install pyswisseph)
  - skyfield    (pip install skyfield)
  - pandas
  - numpy
  - python-dateutil
  - pytz

Install:
```
pip install pyswisseph skyfield pandas numpy python-dateutil pytz
```

Run:
```
python compute_sidereal_planets.py input.csv
```

Output:
- Creates `input.csv.with_astrology.csv` with added columns:
  - For each planet P in [Sun, Moon, Mars, Saturn, Venus]:
    - P_sid_long
    - P_sid_long_over_360
    - P_sid_lat
    - P_zodiac
    - P_zodiac_index
    - P_altitude_deg
    - P_is_above_horizon
    - P_is_combust
    - P_is_retrograde
  - For nodes (both mean and true), for each of Rahu_mean, Ketu_mean, Rahu_true, Ketu_true:
    - <Node>_sid_long
    - <Node>_sid_long_over_360
    - <Node>_sid_lat
    - <Node>_zodiac
    - <Node>_zodiac_index
    - <Node>_altitude_deg (NaN; not computed)
    - <Node>_is_above_horizon (None)
    - <Node>_is_combust
    - <Node>_is_retrograde (None)

Notes:
- The script uses pyswisseph for sidereal ecliptic positions (Lahiri ayanamsha) and skyfield to compute topocentric altitude. Both mean and true nodes are computed with sidereal mode applied.
- If you want to change the combustion threshold or the ayanamsha, edit the top-level constants in `compute_sidereal_planets.py`.
- For Rahu/Ketu altitude we return NaN (nodes are not direct skyfield bodies). You can adapt the code to approximate a topocentric node altitude if needed (extra geometry).
```