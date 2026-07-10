"""Tests for the MCP server wrapper (``astro_engine.mcp_server``).

The server is a thin, JSON-serializable facade over the already-validated
``VedicFeatures`` engine, so these tests check the wiring rather than the
astronomy: the tool registry, datetime parsing, JSON-safety, and that a couple
of calls reproduce the reference chart invariants (Saturn own-sign in Aquarius,
Rahu retrograde) for 2024-01-01 over Delhi.
"""
import asyncio
import json
import os
from datetime import timezone

import pytest

pytest.importorskip("mcp.server.fastmcp")  # skip whole module if the SDK is absent

from .conftest import requires_skyfield

from astro_engine import mcp_server as srv

DELHI = {"latitude": 28.6139, "longitude": 77.2090}
EXPECTED_TOOLS = {
    "vedic_chart", "planet_positions", "ascendant",
    "panchanga", "vimshottari_dasha", "divisional_chart",
}


@pytest.fixture()
def kernel_env(de_kernel, monkeypatch):
    """Point the server's lazy engine at the test kernel."""
    monkeypatch.setenv("ASTRO_KERNEL", de_kernel)
    srv._engine.cache_clear()
    yield de_kernel
    srv._engine.cache_clear()


# -- pure wiring (no ephemeris) ----------------------------------------------

def test_all_tools_registered():
    names = {t.name for t in asyncio.run(srv.mcp.list_tools())}
    assert EXPECTED_TOOLS <= names


def test_parse_dt_naive_is_utc():
    dt = srv._parse_dt("2024-01-01T00:00:00")
    assert dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0


def test_parse_dt_offset_and_z():
    assert srv._parse_dt("2024-01-01T05:30:00+05:30").astimezone(timezone.utc).hour == 0
    assert srv._parse_dt("2024-01-01T00:00:00Z").utcoffset().total_seconds() == 0


def test_parse_dt_rejects_garbage():
    with pytest.raises(ValueError):
        srv._parse_dt("not-a-date")


def test_clean_strips_numpy():
    import numpy as np
    cleaned = srv._clean({"a": np.int64(3), "b": [np.float64(1.5)], "c": np.array([1, 2])})
    assert cleaned == {"a": 3, "b": [1.5], "c": [1, 2]}
    json.dumps(cleaned)  # must not raise


# -- chart invariants (need the ephemeris) -----------------------------------

@requires_skyfield
def test_vedic_chart_reference_invariants(kernel_env):
    c = srv.vedic_chart("2024-01-01T00:00:00", **DELHI)
    assert c["planets"]["Saturn"]["sign"] == "Aquarius"
    assert c["planets"]["Saturn"]["dignity"] == "own"
    assert c["planets"]["Rahu"]["retrograde"] is True
    assert c["panchanga"]["vara"] == "Somavara"  # 2024-01-01 was a Monday
    json.dumps(c)  # end-to-end JSON-serializable


@requires_skyfield
def test_planet_positions_and_ascendant_agree(kernel_env):
    pos = srv.planet_positions("2024-01-01T00:00:00", **DELHI)
    asc = srv.ascendant("2024-01-01T00:00:00", **DELHI)
    assert set(pos) >= {"Sun", "Moon", "Mars", "Saturn", "Rahu", "Ketu"}
    assert asc["sign_index"] == int(asc["longitude"] // 30) % 12


@requires_skyfield
def test_divisional_chart_navamsa_matches_chart(kernel_env):
    d9 = srv.divisional_chart("2024-01-01T00:00:00", varga_number=9, **DELHI)
    chart = srv.vedic_chart("2024-01-01T00:00:00", **DELHI)
    assert d9["varga_number"] == 9
    # the D9 tool must agree with the navamsa the full chart already reports
    for planet, entry in chart["planets"].items():
        assert d9[planet]["sign"] == entry["navamsa"]


@requires_skyfield
def test_vimshottari_levels(kernel_env):
    two = srv.vimshottari_dasha("2024-01-01T00:00:00", levels=2, **DELHI)
    assert two and "sub" in two[0] and "start" in two[0] and "end" in two[0]
