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

def main(): mcp.run()
if __name__ == "__main__": main()
