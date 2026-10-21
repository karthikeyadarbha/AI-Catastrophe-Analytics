"""Planetary conjunction / yuddha (Source Book ch.19, KSS, Brahmagupta KK 1.8, Bhaskara BG).
time of conjunction: t = (A - B)/(a - b) days (a,b daily motions, retrograde negative). Bheda: separation < sum of radii.
Mean angular diameters (Bhaskara II, SiSi 1.10): Mars 4'45", Mercury 6'15", Jupiter 7'20", Venus 9'(OCR-read), Saturn 5'20"."""
DIAM_ARCMIN = {"Mars": 4+45/60, "Mercury": 6+15/60, "Jupiter": 7+20/60, "Venus": 9.0, "Saturn": 5+20/60}
def time_of_conjunction(lon_a, lon_b, v_a, v_b):
    """days until (positive) or since (negative) conjunction in longitude; lon in deg, v in deg/day."""
    d = ((lon_a - lon_b + 180) % 360) - 180
    dv = v_a - v_b
    return None if abs(dv) < 1e-12 else -d/dv
def yuddha(sep_arcmin, p1, p2, lat1_deg=None, lat2_deg=None, rays_state=None):
    """Classify by centre distance. Only Bheda has a quantitative criterion in the source (distance < sum of radii);
    Ullekha, Amsumardana, Apasavya are ordered by increasing distance without stated limits (Brhat Samhita 17.4)."""
    r = 0.5*(DIAM_ARCMIN.get(p1, 0) + DIAM_ARCMIN.get(p2, 0))
    kind = "bheda" if sep_arcmin < r else "beyond-bheda (ullekha/amsumardana/apasavya: limits not given)"
    out = {"type": kind, "sum_of_radii_arcmin": r, "angulas": sep_arcmin/4.0}
    if lat1_deg is not None and lat2_deg is not None:
        if p1 == "Venus" and p2 != "Venus": w = p1
        elif p2 == "Venus" and p1 != "Venus": w = p2
        else: w = p1 if lat1_deg > lat2_deg else p2      # northern planet wins (Brahmagupta/KSS); both-south rule inverted in KSS
        out["victor_by_latitude"] = w
    return out
