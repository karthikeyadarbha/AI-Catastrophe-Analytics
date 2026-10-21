"""Heliacal visibility (Source Book ch.18, 18.5.1-2: degrees of time, kalamsa, from the Sun).
Visible when elongation >= limit: Moon 12, Mars 17, Jupiter 11, Saturn 15, Mercury 13 (direct) / 12 (retro), Venus 9 (direct) / 8 (retro). Same numbers in Aryabhata's midnight system (9,11,13,15,17 for Venus,Jupiter,Mercury,Saturn,Mars).
Approximation: elongation is taken along the ecliptic; the text's dr-kkarma/ayana corrections are not applied."""
LIMIT = {"Moon": (12, 12), "Mars": (17, 17), "Jupiter": (11, 11), "Saturn": (15, 15), "Mercury": (13, 12), "Venus": (9, 8)}
def visible(planet, planet_long, sun_long, retrograde=False):
    el = abs(((planet_long - sun_long + 180) % 360) - 180)
    lim = LIMIT[planet][1 if retrograde else 0]
    return {"visible": el >= lim, "elongation": el, "limit": lim}
