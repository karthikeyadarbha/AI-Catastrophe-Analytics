"""Astro Engine MCP server.

Exposes the validated Vedic feature library (:class:`astro_engine.vedic.VedicFeatures`)
as `Model Context Protocol <https://modelcontextprotocol.io>`_ tools so any
MCP-capable client -- Claude Desktop, Cursor, VS Code Copilot, the Gemini CLI, or
a remote agent built on the Gemini API -- can compute jyotish charts on demand.

Transports
----------
* **stdio** (default) -- for local desktop clients that spawn the server as a
  subprocess (Claude Desktop, Cursor, VS Code, Gemini CLI).
* **streamable-http** -- a remote HTTP server (``--http``) for cloud deployment;
  this is the transport a phone/browser agent reaches over the network.

Run it::

    astro-engine-mcp                 # stdio (local desktop clients)
    astro-engine-mcp --http          # remote HTTP on 127.0.0.1:8000/mcp
    # public, authenticated deployment (recommended for websites / cloud):
    ASTRO_KERNEL=de440.bsp ASTRO_MCP_TOKEN=s3cret \
        astro-engine-mcp --http --host 0.0.0.0 --port 9000 --cors

When ``--token``/``$ASTRO_MCP_TOKEN`` is set, every HTTP request must carry
``Authorization: Bearer <token>``; ``--cors``/``$ASTRO_MCP_CORS`` adds permissive
CORS headers for browser-side MCP clients.

Every tool takes an ISO-8601 datetime (``2024-01-01T05:30:00+05:30`` or a bare
``2024-01-01T00:00:00`` which is read as UTC) plus a latitude/longitude in
decimal degrees, and returns a plain JSON-serializable object.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np

from mcp.server.fastmcp import FastMCP

from .vedic import VedicFeatures, varga
from .vedic import tables as T
from .facade import AstroEngine
from .models.planet import PlanetName

INSTRUCTIONS = (
    "Vedic (sidereal / Lahiri) astrology engine. Point-in-time tools compute a "
    "rasi chart, planetary positions, the ascendant (lagna), the five-limbed "
    "panchanga, the Vimshottari dasha timeline, and arbitrary divisional (varga) "
    "charts. Event tools search over time: find_planetary_event answers "
    "'when did/will a planet get combust / go retrograde / change sign or "
    "nakshatra' (past or future, no date range needed); events_in_range lists "
    "every such event in an explicit window. Datetimes are ISO-8601; a datetime "
    "without an offset is treated as UTC. Positions are sidereal (Lahiri)."
)

mcp = FastMCP("astro-engine", instructions=INSTRUCTIONS)


# --------------------------------------------------------------------------- #
# shared engine (built once; loading the ephemeris sampler is the slow part)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _engine() -> VedicFeatures:
    """Build one :class:`VedicFeatures` from ``$ASTRO_KERNEL`` (or de421.bsp)."""
    return VedicFeatures()


def _parse_dt(datetime_iso: str) -> datetime:
    """Parse an ISO-8601 string to a tz-aware UTC datetime (naive == UTC)."""
    text = datetime_iso.strip().replace("Z", "+00:00").replace("z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse datetime {datetime_iso!r}; expected ISO-8601 "
            "e.g. '2024-01-01T05:30:00+05:30' or '2024-01-01T00:00:00'."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _clean(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays to native JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    return obj


def _chart(datetime_iso: str, latitude: float, longitude: float,
           dasha_levels: int = 1) -> Dict:
    return _clean(_engine().chart(_parse_dt(datetime_iso), latitude, longitude,
                                  dasha_levels=dasha_levels))


# --------------------------------------------------------------------------- #
# event search (combustion / retrograde / ingress) over a time window
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _astro() -> AstroEngine:
    """Build one :class:`AstroEngine` (JPL backend) sharing ``$ASTRO_KERNEL``.

    This is the event-detection facade; it uses the same DE kernel as
    :func:`_engine` so point queries and event scans stay consistent.
    """
    return AstroEngine(backend="jpl",
                       ephemeris=os.environ.get("ASTRO_KERNEL", "de421.bsp"))


#: Friendly event names (and a few Sanskrit/synonym aliases) -> plugin name.
_EVENT_ALIASES = {
    "combustion": "combustion", "combust": "combustion", "asta": "combustion",
    "astangata": "combustion", "astamana": "combustion",
    "retrograde": "retrograde", "retro": "retrograde", "vakri": "retrograde",
    "sign_change": "rasi_transit", "sign": "rasi_transit", "rasi": "rasi_transit",
    "rashi": "rasi_transit", "rasi_transit": "rasi_transit", "ingress": "rasi_transit",
    "nakshatra_change": "nakshatra_transit", "nakshatra": "nakshatra_transit",
    "nakshatra_transit": "nakshatra_transit", "star": "nakshatra_transit",
}

#: Planets each period plugin cannot describe.
_COMBUST_SKIP = {"Sun", "Rahu", "Ketu"}
_RETRO_SKIP = {"Sun", "Moon", "Rahu", "Ketu"}
_TRANSIT_PLUGINS = {"rasi_transit", "nakshatra_transit"}

#: Common Sanskrit / Telugu graha names -> canonical English planet name, so the
#: tool is forgiving if an assistant passes "Budha", "Kuja", "Shani", etc.
_PLANET_ALIASES = {
    "surya": "Sun", "soorya": "Sun", "ravi": "Sun", "aditya": "Sun",
    "chandra": "Moon", "soma": "Moon", "chandrudu": "Moon",
    "mangala": "Mars", "mangal": "Mars", "kuja": "Mars", "angaraka": "Mars",
    "sevvai": "Mars",
    "budha": "Mercury", "budh": "Mercury", "budhudu": "Mercury",
    "guru": "Jupiter", "brihaspati": "Jupiter", "brhaspati": "Jupiter",
    "bruhaspati": "Jupiter",
    "shukra": "Venus", "sukra": "Venus", "shukrudu": "Venus",
    "shani": "Saturn", "sani": "Saturn", "shanaischara": "Saturn",
    "rahu": "Rahu", "ketu": "Ketu",
}

#: Typical days between successive occurrences (a synodic-ish cycle, padded), so
#: the directional search almost always resolves on the first window.
_SPAN_DEFAULTS = {
    "combustion": {"Mercury": 180, "Venus": 780, "Mars": 1000, "Jupiter": 470, "Saturn": 430},
    "retrograde": {"Mercury": 160, "Venus": 650, "Mars": 1000, "Jupiter": 450, "Saturn": 410},
    "rasi_transit": {"Moon": 6, "Sun": 46, "Mercury": 100, "Venus": 180, "Mars": 100,
                     "Jupiter": 470, "Saturn": 1120, "Rahu": 650, "Ketu": 650},
    "nakshatra_transit": {"Moon": 4, "Sun": 20, "Mercury": 40, "Venus": 55, "Mars": 55,
                          "Jupiter": 190, "Saturn": 450, "Rahu": 260, "Ketu": 260},
}
_SPAN_FALLBACK = {"combustion": 1000, "retrograde": 1000,
                  "rasi_transit": 1120, "nakshatra_transit": 450}
_SEARCH_CAP_DAYS = 2200


def _resolve_event_type(event_type: str) -> str:
    key = str(event_type).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return _EVENT_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown event_type {event_type!r}. Use one of: combustion, "
            "retrograde, sign_change, nakshatra_change."
        )


def _coerce_planet_name(planet: str) -> str:
    raw = str(planet).strip()
    alias = _PLANET_ALIASES.get(raw.lower())
    if alias is not None:
        return alias
    try:
        return PlanetName(raw.title()).value
    except ValueError:
        raise ValueError(
            f"Unknown planet {planet!r}. Valid: "
            f"{[p.value for p in PlanetName]}."
        )


def _event_instant(ev: Any) -> datetime:
    """Representative instant of an event: a period's onset, else the instant."""
    date_range = getattr(ev, "date_range", None)
    if date_range is not None:
        return date_range.start.dt
    return ev.date.dt


def _event_to_dict(ev: Any) -> Dict:
    out: Dict[str, Any] = {
        "event_type": ev.event_type,
        "planet": getattr(ev.planet, "value", str(ev.planet)),
    }
    date_range = getattr(ev, "date_range", None)
    if date_range is not None:
        out["start"] = date_range.start.dt.isoformat()
        out["end"] = date_range.end.dt.isoformat()
        out["duration_days"] = round((date_range.end - date_range.start).total_seconds() / 86400.0, 3)
    else:
        out["date"] = ev.date.dt.isoformat()
    extra = getattr(ev, "extra", None)
    if extra:
        out.update(_clean(extra))
    return out


def _default_span_days(plugin: str, planet: str) -> int:
    return int(_SPAN_DEFAULTS.get(plugin, {}).get(planet, _SPAN_FALLBACK.get(plugin, 1000)))


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def vedic_chart(datetime_iso: str, latitude: float, longitude: float,
                dasha_levels: int = 1) -> Dict:
    """Full sidereal (Lahiri) rasi chart for one instant and place.

    Returns the ascendant, every graha (Sun..Ketu + outer planets) with its
    sign, nakshatra + pada, house, retrograde flag, declination, speed, navamsa
    sign and (for the seven classical grahas) dignity; the panchanga; Gulika;
    and the Vimshottari mahadasha timeline.

    Args:
        datetime_iso: ISO-8601 datetime; no offset means UTC.
        latitude: Geographic latitude in decimal degrees (north positive).
        longitude: Geographic longitude in decimal degrees (east positive).
        dasha_levels: 1 = mahadasha only, 2 = maha + antardasha.
    """
    return _chart(datetime_iso, latitude, longitude, dasha_levels=dasha_levels)


@mcp.tool()
def planet_positions(datetime_iso: str, latitude: float,
                     longitude: float) -> Dict:
    """Sidereal positions of every graha for one instant and place.

    Maps each planet to its sign, degree-in-sign, sidereal longitude, nakshatra,
    pada, house (from the ascendant), retrograde flag, declination and speed.

    Args:
        datetime_iso: ISO-8601 datetime; no offset means UTC.
        latitude: Geographic latitude in decimal degrees.
        longitude: Geographic longitude in decimal degrees.
    """
    return _chart(datetime_iso, latitude, longitude)["planets"]


@mcp.tool()
def ascendant(datetime_iso: str, latitude: float, longitude: float) -> Dict:
    """Ascendant (lagna): its sidereal sign, degree-in-sign and longitude.

    Args:
        datetime_iso: ISO-8601 datetime; no offset means UTC.
        latitude: Geographic latitude in decimal degrees.
        longitude: Geographic longitude in decimal degrees.
    """
    return _chart(datetime_iso, latitude, longitude)["ascendant"]


@mcp.tool()
def panchanga(datetime_iso: str, latitude: float, longitude: float) -> Dict:
    """The five limbs (panchanga): tithi, vara, yoga, karana and nakshatra.

    Args:
        datetime_iso: ISO-8601 datetime; no offset means UTC.
        latitude: Geographic latitude in decimal degrees.
        longitude: Geographic longitude in decimal degrees.
    """
    return _chart(datetime_iso, latitude, longitude)["panchanga"]


@mcp.tool()
def vimshottari_dasha(datetime_iso: str, latitude: float, longitude: float,
                      levels: int = 2) -> list:
    """Vimshottari dasha timeline seeded from the Moon's sidereal longitude.

    Args:
        datetime_iso: ISO-8601 datetime; no offset means UTC.
        latitude: Geographic latitude in decimal degrees.
        longitude: Geographic longitude in decimal degrees.
        levels: 1 = mahadasha only, 2 = maha + antardasha (default).
    """
    levels = max(1, min(int(levels), 2))
    return _chart(datetime_iso, latitude, longitude,
                  dasha_levels=levels)["vimshottari"]


@mcp.tool()
def divisional_chart(datetime_iso: str, latitude: float, longitude: float,
                     varga_number: int) -> Dict:
    """Arbitrary divisional (varga) chart -- e.g. D9 navamsa, D10 dasamsa, D12.

    Applies the classical varga division ``varga_number`` to the sidereal
    longitude of the ascendant and every graha, returning each one's varga sign.

    Args:
        datetime_iso: ISO-8601 datetime; no offset means UTC.
        latitude: Geographic latitude in decimal degrees.
        longitude: Geographic longitude in decimal degrees.
        varga_number: Division count N (e.g. 9 for navamsa, 10 for dasamsa).
    """
    n = int(varga_number)
    if n < 1:
        raise ValueError("varga_number must be a positive integer (e.g. 9 for navamsa).")
    chart = _chart(datetime_iso, latitude, longitude)

    def _sign(lon: float) -> Dict:
        idx = int(varga.varga_sign(np.array([float(lon)]), n)[0])
        return {"sign": T.SIGN_NAMES[idx], "sign_index": idx}

    out = {"varga_number": n,
           "Ascendant": _sign(chart["ascendant"]["longitude"])}
    for planet, entry in chart["planets"].items():
        out[planet] = _sign(entry["longitude"])
    return out


@mcp.tool()
def find_planetary_event(event_type: str, planet: str, direction: str = "last",
                         reference_datetime_iso: Optional[str] = None,
                         latitude: float = 0.0, longitude: float = 0.0,
                         max_events: int = 1) -> Dict:
    """Find the most recent ("last") or next ("next") occurrence of an event.

    Answers open-ended natural questions like "when did Mercury last get
    combust (asta)?", "when is Saturn next retrograde?", or "when did Jupiter
    last change sign?" -- WITHOUT the caller supplying a date range. It searches
    outward from a reference instant (default: now), widening the window until
    the event is found.

    Args:
        event_type: "combustion", "retrograde", "sign_change" (rasi ingress) or
            "nakshatra_change" (nakshatra/pada ingress). Synonyms combust/asta,
            retro/vakri, rasi/sign, star/nakshatra are also accepted.
        planet: Graha -- Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu
            or Ketu. Common Sanskrit/Telugu names (Budha, Kuja, Guru, Shukra,
            Shani, Ravi/Surya, Chandra ...) are also accepted. (Combustion
            excludes Sun/Rahu/Ketu; retrograde excludes Sun/Moon/Rahu/Ketu.)
        direction: "last" (most recent past occurrence, default) or "next"
            (nearest future occurrence).
        reference_datetime_iso: ISO-8601 instant to search from; no offset means
            UTC. Defaults to the current time.
        latitude, longitude: Observer location in decimal degrees. Combustion,
            retrograde and ingress are geocentric so location barely matters; it
            is accepted for completeness.
        max_events: How many successive occurrences to return (default 1).

    Returns a dict echoing the resolved query plus an ``events`` list; each event
    has the planet and ISO ``start``/``end`` (periods) or ``date`` (ingress),
    with the entered/left rasi or nakshatra in ``extra`` fields for ingress.
    """
    plugin = _resolve_event_type(event_type)
    planet_name = _coerce_planet_name(planet)
    dir_norm = str(direction).strip().lower()
    if dir_norm not in ("last", "next"):
        raise ValueError("direction must be 'last' or 'next'.")

    base: Dict[str, Any] = {"event_type": plugin, "planet": planet_name,
                            "direction": dir_norm}
    if plugin == "combustion" and planet_name in _COMBUST_SKIP:
        return {**base, "events": [],
                "note": "Combustion (asta) is undefined for the Sun and the "
                        "lunar nodes Rahu/Ketu."}
    if plugin == "retrograde" and planet_name in _RETRO_SKIP:
        return {**base, "events": [],
                "note": "The Sun and Moon never go retrograde; the nodes "
                        "Rahu/Ketu are always retrograde."}

    ref = (_parse_dt(reference_datetime_iso) if reference_datetime_iso
           else datetime.now(timezone.utc))
    span_base = _default_span_days(plugin, planet_name)
    engine = _astro()
    location = (float(latitude), float(longitude))
    want = max(1, int(max_events))

    searched = span_base
    for mult in (1, 3, 9):
        searched = min(span_base * mult, _SEARCH_CAP_DAYS)
        if dir_norm == "last":
            start, end = ref - timedelta(days=searched), ref
        else:
            start, end = ref, ref + timedelta(days=searched)
        repo = engine.find_events(start, end, location,
                                  plugins=[plugin], planets=[planet_name], tz="UTC")
        if dir_norm == "last":
            matches = sorted((e for e in repo if _event_instant(e) <= ref),
                             key=_event_instant, reverse=True)
        else:
            matches = sorted((e for e in repo if _event_instant(e) >= ref),
                             key=_event_instant)
        if matches:
            return {**base, "reference": ref.isoformat(), "searched_days": searched,
                    "events": [_event_to_dict(e) for e in matches[:want]]}
        if searched >= _SEARCH_CAP_DAYS:
            break

    when = "before" if dir_norm == "last" else "after"
    return {**base, "reference": ref.isoformat(), "searched_days": searched, "events": [],
            "note": f"No {plugin.replace('_', ' ')} for {planet_name} {when} "
                    f"{ref.isoformat()} within {searched} days."}


@mcp.tool()
def events_in_range(start_datetime_iso: str, end_datetime_iso: str, event_type: str,
                    latitude: float = 0.0, longitude: float = 0.0,
                    planet: Optional[str] = None) -> Dict:
    """List every occurrence of one event type within an explicit date range.

    Use this for bounded "what happens between X and Y" questions -- e.g. all
    retrograde periods in 2025, or Jupiter's sign changes this decade. For
    open-ended "when did/will ..." questions prefer :func:`find_planetary_event`.

    Args:
        start_datetime_iso, end_datetime_iso: ISO-8601 bounds; no offset = UTC.
        event_type: combustion, retrograde, sign_change or nakshatra_change.
        latitude, longitude: observer location (geocentric events; barely matters).
        planet: optional single graha to restrict to; omit for all nine.

    To protect a shared/free host the span is capped (the fast-moving Moon makes
    ingress scans expensive): about 150 days for sign/nakshatra ingress when the
    Moon is included, ~11 years for a single non-Moon planet's ingress, and
    ~6 years for combustion/retrograde (less when scanning all planets).
    """
    plugin = _resolve_event_type(event_type)
    start = _parse_dt(start_datetime_iso)
    end = _parse_dt(end_datetime_iso)
    if end <= start:
        raise ValueError("end_datetime_iso must be after start_datetime_iso.")

    planet_name = _coerce_planet_name(planet) if planet else None
    span_days = (end - start).total_seconds() / 86400.0
    if plugin in _TRANSIT_PLUGINS:
        moon_involved = planet_name is None or planet_name == "Moon"
        cap = 150 if moon_involved else 4000
    else:
        cap = 2200 if planet_name else 800
    if span_days > cap:
        raise ValueError(
            f"Requested {span_days:.0f}-day range exceeds the {cap}-day cap for "
            f"{plugin!r}. Narrow the range or query a single planet at a time."
        )

    engine = _astro()
    repo = engine.find_events(start, end, (float(latitude), float(longitude)),
                              plugins=[plugin],
                              planets=[planet_name] if planet_name else None,
                              tz="UTC")
    events = sorted(repo, key=_event_instant)
    return {"event_type": plugin, "planet": planet_name or "all",
            "start": start.isoformat(), "end": end.isoformat(),
            "count": len(events), "events": [_event_to_dict(e) for e in events]}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="astro-engine-mcp",
        description="Serve the Astro Engine Vedic library over the Model Context Protocol.",
    )
    p.add_argument("--transport", choices=["stdio", "streamable-http", "sse"],
                   default=os.environ.get("MCP_TRANSPORT", "stdio"),
                   help="MCP transport (default: stdio, or $MCP_TRANSPORT).")
    p.add_argument("--http", action="store_true",
                   help="Shorthand for --transport streamable-http (remote server).")
    p.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"),
                   help="Bind host for HTTP transports (default: 127.0.0.1 / $MCP_HOST).")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("PORT") or os.environ.get("MCP_PORT") or "8000"),
                   help="Bind port for HTTP transports (default: $PORT / $MCP_PORT / 8000). "
                        "$PORT lets one image run unchanged on HF Spaces, Cloud Run, Render, etc.")
    p.add_argument("--token", default=os.environ.get("ASTRO_MCP_TOKEN"),
                   help="Require 'Authorization: Bearer <token>' on HTTP requests "
                        "(default: $ASTRO_MCP_TOKEN). Strongly recommended for any "
                        "publicly reachable deployment.")
    p.add_argument("--cors", action="store_true",
                   default=os.environ.get("ASTRO_MCP_CORS", "").lower() in ("1", "true", "yes"),
                   help="Send permissive CORS headers (needed only for browser-side "
                        "MCP clients; server-to-server callers do not need this).")
    p.add_argument("--stateless", action="store_true",
                   default=os.environ.get("ASTRO_MCP_STATELESS", "").lower() in ("1", "true", "yes"),
                   help="Stateless JSON HTTP mode (no server-side sessions or SSE stream). "
                        "Recommended behind serverless / free-tier proxies that buffer or "
                        "recycle connections (Hugging Face Spaces, Cloud Run, Render).")
    return p


def _run_http(transport: str, host: str, port: int, token: str | None,
              cors: bool) -> None:
    """Run an HTTP transport, optionally guarded by a bearer token / CORS.

    Without a token we defer to the SDK's own runner. With a token we wrap the
    ASGI app in a tiny middleware so a public deployment is not wide open.
    """
    mcp.settings.host = host
    mcp.settings.port = port
    if not token and not cors:
        mcp.run(transport=transport)
        return

    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse, Response

    app = mcp.sse_app() if transport == "sse" else mcp.streamable_http_app()

    if token:
        class _BearerAuth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.method == "OPTIONS":            # let CORS preflight through
                    return await call_next(request)
                if request.headers.get("authorization") != f"Bearer {token}":
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
                return await call_next(request)

        app.add_middleware(_BearerAuth)

    if cors:
        from starlette.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"],
                           expose_headers=["Mcp-Session-Id"])

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    args = _build_arg_parser().parse_args()
    transport = "streamable-http" if args.http else args.transport
    if transport in ("streamable-http", "sse"):
        if args.stateless:
            mcp.settings.stateless_http = True
            mcp.settings.json_response = True
        _run_http(transport, args.host, args.port, args.token, args.cors)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
