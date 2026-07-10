"""Ascendant (Lagna) computation: structure, backend parity and known values.

The Ascendant is validated against Swiss Ephemeris (which drikpanchang.com uses
internally) as ground truth. The JPL backend must track Swiss closely across a
wide date range -- including far-past and far-future dates, where a naive
sidereal-time model would diverge by many arcminutes.
"""
import pytest

from astro_engine.models.zodiac_sign import ZodiacSign
from astro_engine.models.nakshatra import Nakshatra

from .conftest import requires_swiss, requires_skyfield

LOC = (17.385, 78.4867)  # Hyderabad
TZ = "Asia/Kolkata"

# (label, local_datetime) spanning recent, far-past and far-future.
CASES = [
    ("recent", "2025-01-01 06:00:00"),
    ("recent-noon", "2025-06-15 12:30:00"),
    ("past", "1900-03-21 05:45:00"),
    ("future", "2099-11-09 23:15:00"),
    ("far-future", "2200-07-04 18:00:00"),
]

# Swiss-Ephemeris reference longitudes (deg) for the cases above; Swiss is the
# drikpanchang-equivalent ground truth. Guards the Swiss path against regressions.
SWISS_REFERENCE = {
    "2025-01-01 06:00:00": 244.9746,
    "2025-06-15 12:30:00": 153.2026,
    "1900-03-21 05:45:00": 328.1102,
    "2099-11-09 23:15:00": 102.2777,
    "2200-07-04 18:00:00": 243.8840,
}


def _check_ascendant(asc):
    assert 0.0 <= asc.longitude < 360.0
    assert isinstance(asc.rasi, ZodiacSign)
    assert isinstance(asc.nakshatra, Nakshatra)
    assert 1 <= asc.pada <= 4
    assert asc.dms  # non-empty formatted string


@requires_swiss
@pytest.mark.swiss
class TestSwissLagna:
    def test_structure(self, swiss_engine):
        for _, when in CASES:
            _check_ascendant(swiss_engine.lagna(when, LOC, tz=TZ))

    def test_ascendant_alias(self, swiss_engine):
        a = swiss_engine.lagna("2025-01-01 06:00:00", LOC, tz=TZ)
        b = swiss_engine.ascendant("2025-01-01 06:00:00", LOC, tz=TZ)
        assert a.longitude == b.longitude

    def test_known_reference_values(self, swiss_engine):
        for when, expected in SWISS_REFERENCE.items():
            got = swiss_engine.lagna(when, LOC, tz=TZ).longitude
            d = abs(((got - expected + 180.0) % 360.0) - 180.0)
            assert d < 0.01, f"{when}: got {got:.4f}, expected {expected:.4f}"

    def test_longitude_only_matches_model(self, swiss_engine):
        when = "2025-06-15 12:30:00"
        lon = swiss_engine.ascendant_longitude(when, LOC, tz=TZ)
        asc = swiss_engine.lagna(when, LOC, tz=TZ)
        assert abs(lon - asc.longitude) < 1e-9


@requires_skyfield
@pytest.mark.jpl
class TestJplLagna:
    def test_structure(self, jpl_engine):
        for _, when in CASES:
            _check_ascendant(jpl_engine.lagna(when, LOC, tz=TZ))


# --- drikpanchang.com ground truth ---------------------------------------
# Rising sign (Janma Lagna) collected from drikpanchang's Lagna Calculator
# (Lahiri / Chitra-Paksha ayanamsha) for Hyderabad as geocoded BY drikpanchang.
# Spans very-past, recent and very-future. 0=Aries .. 11=Pisces.
DRIK_LOC = (17.3842, 78.4564)  # Hyderabad, exactly as drikpanchang resolves it
DRIK_CASES = [
    ("very-past", "1900-03-21 05:45:00", 10),  # Aquarius  (Kumbha)
    ("recent", "2025-01-01 06:00:00", 8),      # Sagittarius (Dhanu)
    ("very-future", "2076-06-15 12:30:00", 5),  # Virgo (Kanya)
]


@requires_swiss
@pytest.mark.swiss
class TestDrikpanchangGroundTruth:
    """Rising sign must match drikpanchang.com across past/recent/future."""

    def test_swiss_signs_match_drikpanchang(self, swiss_engine):
        for label, when, sign_idx in DRIK_CASES:
            asc = swiss_engine.lagna(when, DRIK_LOC, tz=TZ)
            got = int(asc.longitude // 30) % 12
            assert got == sign_idx, f"{label} {when}: sign {got} != drik {sign_idx}"


@requires_skyfield
@pytest.mark.jpl
class TestDrikpanchangGroundTruthJpl:
    """Independent JPL backend must also match drikpanchang's rising sign."""

    def test_jpl_signs_match_drikpanchang(self, jpl_engine):
        for label, when, sign_idx in DRIK_CASES:
            asc = jpl_engine.lagna(when, DRIK_LOC, tz=TZ)
            got = int(asc.longitude // 30) % 12
            assert got == sign_idx, f"{label} {when}: sign {got} != drik {sign_idx}"


@requires_swiss
@requires_skyfield
@pytest.mark.jpl
@pytest.mark.swiss
class TestLagnaParity:
    """JPL Ascendant must match Swiss across recent, past and future dates.

    The sidereal-time model is derived from the UT Julian Day (as Swiss does),
    which keeps far-future dates aligned; a ``t.gast``-based model would drift by
    tens of arcminutes by 2200.
    """

    def test_parity_across_epochs(self, swiss_engine, jpl_engine):
        worst = 0.0
        for label, when in CASES:
            a = swiss_engine.lagna(when, LOC, tz=TZ)
            b = jpl_engine.lagna(when, LOC, tz=TZ)
            d = abs(((a.longitude - b.longitude + 180.0) % 360.0) - 180.0) * 3600.0
            worst = max(worst, d)
            # Rasi (rising sign) must always agree.
            assert a.rasi == b.rasi, f"{label}: swiss={a.rasi} jpl={b.rasi}"
        # Empirically <=16" over 1900-2200; allow a comfortable 60" margin.
        assert worst < 60.0, f"max Ascendant divergence {worst:.1f}\""
