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
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict

import numpy as np

from mcp.server.fastmcp import FastMCP

from .vedic import VedicFeatures, varga
from .vedic import tables as T

INSTRUCTIONS = (
    "Vedic (sidereal / Lahiri) astrology engine. Tools compute a rasi chart, "
    "planetary positions, the ascendant (lagna), the five-limbed panchanga, the "
    "Vimshottari dasha timeline, and arbitrary divisional (varga) charts for a "
    "given UTC/offset datetime and geographic location. Datetimes are ISO-8601; "
    "a datetime without an offset is treated as UTC. Positions are sidereal."
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
