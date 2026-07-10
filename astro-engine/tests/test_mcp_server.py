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
from datetime import datetime, timezone

import pytest

pytest.importorskip("mcp.server.fastmcp")  # skip whole module if the SDK is absent

from .conftest import requires_skyfield

from astro_engine import mcp_server as srv

DELHI = {"latitude": 28.6139, "longitude": 77.2090}
EXPECTED_TOOLS = {
    "vedic_chart", "planet_positions", "ascendant",
    "panchanga", "vimshottari_dasha", "divisional_chart",
    "find_planetary_event", "events_in_range",
}


@pytest.fixture()
def kernel_env(de_kernel, monkeypatch):
    """Point the server's lazy engines at the test kernel."""
    monkeypatch.setenv("ASTRO_KERNEL", de_kernel)
    srv._engine.cache_clear()
    srv._astro.cache_clear()
    yield de_kernel
    srv._engine.cache_clear()
    srv._astro.cache_clear()


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


# -- event search: pure wiring (no ephemeris) --------------------------------

def test_resolve_event_type_aliases():
    assert srv._resolve_event_type("combust") == "combustion"
    assert srv._resolve_event_type("asta") == "combustion"
    assert srv._resolve_event_type("vakri") == "retrograde"
    assert srv._resolve_event_type("Sign Change") == "rasi_transit"
    assert srv._resolve_event_type("nakshatra") == "nakshatra_transit"
    with pytest.raises(ValueError):
        srv._resolve_event_type("wobble")


def test_coerce_planet_name_sanskrit_aliases():
    assert srv._coerce_planet_name("Budha") == "Mercury"
    assert srv._coerce_planet_name("kuja") == "Mars"
    assert srv._coerce_planet_name("GURU") == "Jupiter"
    assert srv._coerce_planet_name("shani") == "Saturn"
    assert srv._coerce_planet_name("mercury") == "Mercury"
    with pytest.raises(ValueError):
        srv._coerce_planet_name("Nibiru")


def test_find_event_guards_return_note_without_ephemeris():
    # Sun/Rahu/Ketu combustion and Sun/Moon retrograde are undefined; these
    # short-circuit before any ephemeris call.
    sun = srv.find_planetary_event("combustion", "Sun", "last",
                                   reference_datetime_iso="2024-01-01T00:00:00")
    assert sun["events"] == [] and "note" in sun
    moon = srv.find_planetary_event("retrograde", "Moon", "next",
                                    reference_datetime_iso="2024-01-01T00:00:00")
    assert moon["events"] == [] and "note" in moon


def test_events_in_range_rejects_bad_bounds():
    with pytest.raises(ValueError):
        srv.events_in_range("2024-06-01T00:00:00", "2024-01-01T00:00:00", "retrograde")


def test_events_in_range_enforces_cost_cap():
    # 24 years of nakshatra ingress (Moon included) is far over the 150-day cap.
    with pytest.raises(ValueError):
        srv.events_in_range("2000-01-01T00:00:00", "2024-01-01T00:00:00",
                            "nakshatra_change")


# -- event search: against the ephemeris -------------------------------------

@requires_skyfield
def test_find_last_combustion_before_reference(kernel_env):
    ref = "2024-06-01T00:00:00"
    r = srv.find_planetary_event("asta", "Budha", "last", reference_datetime_iso=ref)
    assert r["event_type"] == "combustion" and r["events"]
    ev = r["events"][0]
    assert ev["planet"] == "Mercury"
    # the onset must not be in the future relative to the reference instant
    assert datetime.fromisoformat(ev["start"]) <= datetime.fromisoformat(ref).replace(tzinfo=timezone.utc)
    json.dumps(r)


@requires_skyfield
def test_find_next_retrograde_is_future(kernel_env):
    ref = datetime.fromisoformat("2024-06-01T00:00:00").replace(tzinfo=timezone.utc)
    r = srv.find_planetary_event("retrograde", "Mercury", "next",
                                 reference_datetime_iso="2024-06-01T00:00:00")
    assert r["events"]
    ev = r["events"][0]
    assert ev["planet"] == "Mercury"
    # Mercury's next retrograde after 2024-06-01 stationed on 2024-08-05
    assert datetime.fromisoformat(ev["start"]) >= ref
    assert datetime.fromisoformat(ev["start"]).month == 8


@requires_skyfield
def test_events_in_range_counts_mercury_retrogrades_2024(kernel_env):
    r = srv.events_in_range("2024-01-01T00:00:00", "2024-12-31T00:00:00",
                            "retrograde", planet="Mercury")
    # Mercury retrogrades ~3x/yr; with the plugin closing an open period at the
    # boundary we expect at least three periods in 2024.
    assert r["count"] >= 3
    assert all(e["planet"] == "Mercury" for e in r["events"])
    assert r["events"] == sorted(r["events"], key=lambda e: e["start"])


@requires_skyfield
def test_find_last_sign_change_reports_ingress(kernel_env):
    r = srv.find_planetary_event("sign_change", "Jupiter", "last",
                                 reference_datetime_iso="2024-06-01T00:00:00")
    assert r["events"]
    ev = r["events"][0]
    # Jupiter entered Vrishabam (Taurus) on 2024-05-01, leaving Mesham (Aries)
    assert ev["planet"] == "Jupiter" and "date" in ev
    assert "rasi_entered" in ev and "rasi_left" in ev

