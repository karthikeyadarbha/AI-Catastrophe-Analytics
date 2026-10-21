"""
Constants for Astral Ephemeris and Astrology engine.
"""
from datetime import timezone, timedelta
import swisseph as swe

# Timezones
UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30), name="IST")
DEFAULT_TIMEZONE = UTC

# Swiss Ephemeris Planet Codes
SWISSEPH_PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE,
    "True_Rahu": swe.TRUE_NODE,
    "Ketu": "KETU",  # 180° opposition to Rahu
    "True_Ketu": "TRUE_KETU",
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO,
}

# 7 Traditional Chara Karaka Eligible Grahas (Jaimini system)
CHARA_GRAHAS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
# 8-Karaka variant includes Rahu
CHARA_GRAHAS_8 = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu"]

CHARA_KARAKA_NAMES = [
    "Atmakaraka",      # AK - 1st highest degree
    "Amatyakaraka",    # AmK - 2nd
    "Bhratrukaraka",   # BK - 3rd
    "Matrukaraka",     # MK - 4th
    "Pitrukaraka",     # PK - 5th
    "Jnatikaraka",     # GK - 6th
    "Darakaraka",      # DK - 7th
]

CHARA_KARAKA_CODES = ["AK", "AmK", "BK", "MK", "PK", "GK", "DK"]

# Zodiac Signs (Sanskrit / Western)
ZODIAC_SIGNS = [
    "Mesham", "Vrishabam", "Mithunam", "Katakam",
    "Simham", "Kanya", "Tula", "Vrishchikam",
    "Dhanassu", "Makaram", "Kumbham", "Meenam"
]

WESTERN_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# 27 Nakshatras
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashirsha", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

# Combustion Limits (degrees from Sun)
COMBUSTION_LIMITS = {
    "Moon":    {"direct": 12.0, "retro": 12.0},
    "Mars":    {"direct": 17.0, "retro": 17.0},
    "Mercury": {"direct": 14.0, "retro": 12.0},
    "Jupiter": {"direct": 11.0, "retro": 11.0},
    "Venus":   {"direct": 10.0, "retro": 8.0},
    "Saturn":  {"direct": 15.0, "retro": 15.0},
}

# Ayanamsa Options
AYANAMSA_MODES = {
    "lahiri": swe.SIDM_LAHIRI,
    "raman": swe.SIDM_RAMAN,
    "kp": swe.SIDM_KRISHNAMURTI,
    "fagan_bradley": swe.SIDM_FAGAN_BRADLEY,
    "yukteshwar": swe.SIDM_YUKTESHWAR,
    "sayana": -1,  # Tropical (no ayanamsa)
}

DEFAULT_AYANAMSA = "lahiri"
