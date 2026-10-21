"""indic-mcp-server: sibling MCP server to astral-ephemeris. Engine 1 (rules) + Engine 2 (computation)."""
import json
try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:                       # mcp 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP
from . import __version__
from .rules import list_sources as _ls, get_rule as _gr, search_rules as _sr
from .compute import planet_positions, jya as _jya, capa as _capa, ahargana
from .compute import series as _series
from .compute import ayanamsa as _ay, panchanga as _pc, latitude as _lt, conjunction as _cj, visibility as _vs, occultation as _oc

mcp = FastMCP("indic-engine")

@mcp.tool()
def list_sources() -> str:
    """List the treatises in the registry with rule counts (implemented vs registered-only)."""
    return json.dumps({"version": __version__, "sources": _ls()}, ensure_ascii=False)

@mcp.tool()
def search_rules(query: str = "", text: str = "", status: str = "") -> str:
    """Search verse-referenced rules. text: Tantrasangraha | Aryabhatiya | Ganita-yukti-bhasa | Cross-text. status: implemented | registered-only."""
    return json.dumps(_sr(query, text or None, status or None), ensure_ascii=False)

@mcp.tool()
def get_rule(rule_id: str) -> str:
    """Full record for a rule id such as TS.2.26.exterior, with reference, formula, Sanskrit (if verified) and notes."""
    return json.dumps(_gr(rule_id), ensure_ascii=False)

@mcp.tool()
def indic_positions(jd_ut: float, param: str = "TS", scheme: str = "madhava") -> str:
    """Sidereal true longitudes (degrees, Indian sidereal frame) of Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn and mean longitudes, by Indian methods.
    param: TS (Tantrasangraha parameters, primary) | AB (Aryabhatiya parameters, lower confidence). scheme: aryabhata|tantrasangraha|madhava|series|exact."""
    r = planet_positions(jd_ut, param, scheme)
    return json.dumps({"engine_version": __version__, **r}, default=float)

@mcp.tool()
def jya(arc_minutes: float, scheme: str = "madhava") -> str:
    """Rsine (arc in minutes) under the chosen Indian table or series."""
    return json.dumps({"jya": _jya(arc_minutes, scheme), "scheme": scheme})

@mcp.tool()
def madhava_series(kind: str, value: float) -> str:
    """kind: sine|cosine (value = arc in minutes) or arctan (value = jya/kotijya in [-1,1]; end correction applied)."""
    if kind == "sine": out = _series.madhava_sine(value)
    elif kind == "cosine": out = _series.madhava_cosine(value)
    elif kind == "arctan": out = _series.madhava_arctan_corrected(value)
    else: out = None
    return json.dumps({"kind": kind, "value": out})

@mcp.tool()
def compare_treatises(jd_ut: float, scheme: str = "madhava") -> str:
    """Compute the sky with both parameter sets (TS and AB) and report the difference per body: how the texts complement each other."""
    a = planet_positions(jd_ut, "TS", scheme)["true"]; b = planet_positions(jd_ut, "AB", scheme)["true"]
    diff = {k: ((b[k] - a[k] + 180) % 360) - 180 for k in a}
    return json.dumps({"TS": a, "AB": b, "AB_minus_TS_deg": diff}, default=float)

@mcp.tool()
def ayanamsa(jd: float, model: str = "Lahiri", true_nutation: bool = False) -> str:
    """Ayanamsa in degrees. Models: Tropical, Lahiri, Raman (BVRaman), KP (Krishnamurti), ChandraHari, Dulakara (Indrasena), and treatise models SuryaNilakantha, AryabhataSchool, DevaAmaraja."""
    return json.dumps({"model": model, "ayanamsa_deg": _ay.ayanamsa(jd, model, true_nutation), "models": _ay.models()})

@mcp.tool()
def compare_ayanamsas(jd: float) -> str:
    """Ayanamsa of every model at one Julian day (degrees)."""
    return json.dumps({m: _ay.ayanamsa(jd, m) for m in _ay.models()})

@mcp.tool()
def panchanga(jd_ut: float, ayanamsa_model: str = "Lahiri") -> str:
    """Tithi, nakshatra, yoga, karana from the Indic engine's Sun and Moon; vyatipata/vaidhrta distance using the chosen ayanamsa."""
    t = planet_positions(jd_ut)["true"]
    return json.dumps(_pc.panchanga(t["Sun"], t["Moon"], _ay.ayanamsa(jd_ut, ayanamsa_model)))

@mcp.tool()
def planet_latitudes(jd_ut: float) -> str:
    """Latitudes (degrees, north positive) of the Moon and five planets by the Aryabhata/Bhaskara rule."""
    t = planet_positions(jd_ut)["true"]
    out = {p: _lt.planet_latitude(p, jd_ut) for p in ("Mars", "Mercury", "Jupiter", "Venus", "Saturn")}
    out["Moon"] = _lt.moon_latitude(t["Moon"], t["MoonNode"])
    return json.dumps(out)

@mcp.tool()
def graha_yuddha(jd_ut: float, planet_a: str, planet_b: str) -> str:
    """Longitude/latitude separation of two planets, bheda test (distance < sum of radii), victor by latitude."""
    t = planet_positions(jd_ut)["true"]; la = _lt.planet_latitude(planet_a, jd_ut); lb = _lt.planet_latitude(planet_b, jd_ut)
    dl = ((t[planet_a] - t[planet_b] + 180) % 360) - 180
    sep = (dl**2 + (la - lb)**2) ** 0.5 * 60.0
    return json.dumps({"longitude_diff_deg": dl, "separation_arcmin": sep, **_cj.yuddha(sep, planet_a, planet_b, la, lb)})

@mcp.tool()
def visibility(jd_ut: float, planet: str) -> str:
    """Heliacal visibility test (Source Book 18.5.1 limits), ecliptic elongation only."""
    t = planet_positions(jd_ut)["true"]
    ph = planet_positions(jd_ut + 0.5)["true"][planet]; retro = ((ph - t[planet] + 180) % 360) - 180 < 0
    return json.dumps(_vs.visible(planet, t[planet], t["Sun"], retro))

@mcp.tool()
def rohini_bheda(jd_ut: float, planet: str, window_deg: float = 1.0, rule: str = "brahmagupta") -> str:
    """Rohini-sakata-bheda test: planet at 17 deg Vrsabha (47 deg) with southern latitude > 2 deg (Brahmagupta) or 50 angulas (Ganesa)."""
    t = planet_positions(jd_ut)["true"]
    lat = _lt.moon_latitude(t["Moon"], t["MoonNode"]) if planet == "Moon" else _lt.planet_latitude(planet, jd_ut)
    return json.dumps({"longitude": t[planet], "latitude": lat, "bheda": _oc.rohini_sakata_bheda(t[planet], lat, window_deg, rule)})

def main(): mcp.run()
if __name__ == "__main__": main()
