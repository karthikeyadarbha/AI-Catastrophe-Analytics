"""Star occultation and the Rohini wain (Source Book ch.20: Brahmagupta KK 1.9.14-16; SaSi 8.13-14; Ganesa GL 11.7-8; Bhaskara I LBh 8.)"""
JUNCTION_LAT_DEG = {"Asvini": +10, "Bharani": +12, "Krttika": +5, "Rohini": -5, "Mrgasira": -10, "Ardra": -11, "Punarvasu": +6, "Pusya": 0, "Aslesa": -7, "Magha": 0, "Purvaphalguni": +12, "Uttaraphalguni": +13, "Hasta": -11, "Citra": -2, "Svati": +37, "Visakha": -1.5, "Anuradha": -3, "Jyestha": -4, "Mula": -8.5, "Purvasadha": -5.333, "Uttarasadha": -5, "Sravana": +30/60*0+0.5, "Abhijit": +62, "Dhanistha": +36, "Satabhisaj": -0.333, "Purvabhadrapada": +24, "Uttarabhadrapada": +26, "Revati": 0}
# NOTE: values are the Source Book ch.12.6 lists as read from the OCR; the listing mixes degrees/minutes per source (Lalla vs Bhaskara I). Used only for Rohini here.
ROHINI_LONG_DEG = 49.3   # polar longitude 1s 19d 18m (Source Book 12.5, OCR-read)
def occults(planet_lat_deg, star_lat_deg, planet_semidiam_deg):
    """Brahmagupta KK 1.9.14: when planet and star share a longitude, the planet occults the star if their latitudes differ by less than the planet's semi-diameter."""
    return abs(planet_lat_deg - star_lat_deg) < planet_semidiam_deg
def rohini_sakata_bheda(planet_long_deg, planet_lat_deg, window_deg=1.0, rule="brahmagupta"):
    """KK 1.9.15: longitude 1 sign 17 deg (=47 deg, after dr-karma) and southern latitude > 2 deg.
    GL 11.7: same longitude, southern latitude > 50 angulas = 2 deg 5'. window_deg is OUR tolerance (the text states no window)."""
    thr = 2.0 if rule == "brahmagupta" else 50/24.0
    return abs(planet_long_deg - 47.0) <= window_deg and planet_lat_deg < -thr
