"""Shared classical tables for the Vedic sub-libraries (0 = Aries ... 11 = Pisces)."""
from __future__ import annotations

from typing import Dict, List

SIGN_NAMES: List[str] = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

NAKSHATRA_NAMES: List[str] = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "PurvaPhalguni", "UttaraPhalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "PurvaAshadha", "UttaraAshadha", "Shravana", "Dhanishta",
    "Shatabhisha", "PurvaBhadrapada", "UttaraBhadrapada", "Revati",
]

TITHI_NAMES: List[str] = [
    "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi",
    "Saptami", "Ashtami", "Navami", "Dashami", "Ekadashi", "Dwadashi",
    "Trayodashi", "Chaturdashi", "Purnima/Amavasya",
]

YOGA_NAMES: List[str] = [
    "Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda",
    "Sukarma", "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva", "Vyaghata",
    "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyana", "Parigha", "Shiva",
    "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Indra", "Vaidhriti",
]

KARANA_NAMES: List[str] = [
    "Kimstughna", "Bava", "Balava", "Kaulava", "Taitila", "Gara", "Vanija",
    "Vishti", "Shakuni", "Chatushpada", "Naga",
]
_MOVABLE_KARANA = ["Bava", "Balava", "Kaulava", "Taitila", "Gara", "Vanija", "Vishti"]

VARA_NAMES: List[str] = [
    "Ravivara", "Somavara", "Mangalavara", "Budhavara",
    "Guruvara", "Shukravara", "Shanivara",
]  # indexed in astrological weekday order Sun=0 .. Sat=6

#: Sign ruler (graha name) for each of the 12 signs.
SIGN_LORDS: List[str] = [
    "Mars", "Venus", "Mercury", "Moon", "Sun", "Mercury",
    "Venus", "Mars", "Jupiter", "Saturn", "Saturn", "Jupiter",
]

#: Exaltation sign index per planet (nodes omitted -- traditions disagree).
EXALTATION: Dict[str, int] = {
    "Sun": 0, "Moon": 1, "Mars": 9, "Mercury": 5,
    "Jupiter": 3, "Venus": 11, "Saturn": 6,
}
#: Debilitation = exaltation sign + 6 (the opposite sign).
DEBILITATION: Dict[str, int] = {p: (s + 6) % 12 for p, s in EXALTATION.items()}

OWN_SIGNS: Dict[str, List[int]] = {
    "Sun": [4], "Moon": [3], "Mars": [0, 7], "Mercury": [2, 5],
    "Jupiter": [8, 11], "Venus": [1, 6], "Saturn": [9, 10],
}
MOOLATRIKONA: Dict[str, int] = {
    "Sun": 4, "Moon": 1, "Mars": 0, "Mercury": 5,
    "Jupiter": 8, "Venus": 6, "Saturn": 10,
}

#: Natural friendships (classical). Planets not listed as friend/enemy are neutral.
NATURAL_FRIENDS: Dict[str, set] = {
    "Sun": {"Moon", "Mars", "Jupiter"},
    "Moon": {"Sun", "Mercury"},
    "Mars": {"Sun", "Moon", "Jupiter"},
    "Mercury": {"Sun", "Venus"},
    "Jupiter": {"Sun", "Moon", "Mars"},
    "Venus": {"Mercury", "Saturn"},
    "Saturn": {"Mercury", "Venus"},
}
NATURAL_ENEMIES: Dict[str, set] = {
    "Sun": {"Venus", "Saturn"},
    "Moon": set(),
    "Mars": {"Mercury"},
    "Mercury": {"Moon"},
    "Jupiter": {"Mercury", "Venus"},
    "Venus": {"Sun", "Moon"},
    "Saturn": {"Sun", "Moon", "Mars"},
}

#: Vimshottari nakshatra-lord order (repeats 3x across the 27 nakshatras).
NAKSHATRA_LORD_ORDER: List[str] = [
    "Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury",
]
VIMSHOTTARI_YEARS: Dict[str, float] = {
    "Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7,
    "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17,
}
VIMSHOTTARI_TOTAL = 120.0

#: Weekday lord in ASTROLOGICAL order Sun=0, Mon=1, ... Sat=6.
WEEKDAY_LORDS: List[str] = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]


def nakshatra_lord(nak_index) -> List[str]:
    """Vimshottari lord(s) for nakshatra index/array (0-based)."""
    import numpy as np
    order = np.array(NAKSHATRA_LORD_ORDER)
    return order[np.asarray(nak_index) % 9]


def karana_name_index(half_tithi):
    """Map a half-tithi index (0-59) to a karana index into ``KARANA_NAMES``."""
    import numpy as np
    h = np.atleast_1d(np.asarray(half_tithi)) % 60
    out = np.zeros(h.shape, dtype=int)
    # 0 -> Kimstughna(0); 57,58,59 -> Shakuni(8),Chatushpada(9),Naga(10)
    mid = (h >= 1) & (h <= 56)
    out[mid] = 1 + ((h[mid] - 1) % 7)  # movable karanas -> indices 1..7
    out[h == 57] = 8
    out[h == 58] = 9
    out[h == 59] = 10
    return out
