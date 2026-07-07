"""Tests for the vectorized Vedic feature sub-libraries (``astro_engine.vedic``).

The substrate is cross-checked against the *scalar* :class:`AstroEngine` (the
independently-validated reference); the derived features are checked against
closed-form rules and a hand-verified reference chart.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from .conftest import requires_skyfield

pytestmark = requires_skyfield


# -- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def sampler(de_kernel):
    from astro_engine.vedic import SkySampler
    return SkySampler(de_kernel)


@pytest.fixture(scope="module")
def vf(de_kernel):
    from astro_engine.vedic import VedicFeatures
    return VedicFeatures(de_kernel)


DELHI = (28.6139, 77.2090)
WHEN = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


# -- substrate parity vs the scalar engine -----------------------------------

@requires_skyfield
def test_sidereal_longitude_matches_scalar(sampler, jpl_engine):
    bodies = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
    dates = [datetime(1950, 3, 21, 6, tzinfo=timezone.utc),
             datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
             datetime(2050, 9, 9, 18, tzinfo=timezone.utc)]  # within de421 (1899-2053)
    for when in dates:
        s = sampler.sample([when], *DELHI)
        for b in bodies:
            got = float(s.sid_lon[b][0])
            ref = jpl_engine.longitude(b, when, DELHI)
            diff = abs((got - ref + 180) % 360 - 180)
            assert diff < 1e-4, f"{b} @ {when}: {got} vs {ref} (diff {diff})"


@requires_skyfield
def test_ascendant_matches_scalar(sampler, jpl_engine):
    for when in [datetime(1975, 7, 4, 3, tzinfo=timezone.utc),
                 datetime(2024, 1, 1, 12, tzinfo=timezone.utc)]:
        s = sampler.sample([when], *DELHI)
        got = float(s.asc_sid[0])
        ref = jpl_engine.ascendant_longitude(when, DELHI)
        diff = abs((got - ref + 180) % 360 - 180)
        assert diff < 5e-3, f"asc @ {when}: {got} vs {ref} (diff {diff})"


# -- panchanga against a hand-verified reference chart -----------------------

@requires_skyfield
def test_panchanga_reference_chart(vf):
    c = vf.chart(WHEN, *DELHI)
    assert c["panchanga"]["vara"] == "Somavara"                 # 2024-01-01 = Monday
    assert c["panchanga"]["tithi"].startswith("Krishna")        # ~6d after full moon
    assert c["panchanga"]["nakshatra"] == "PurvaPhalguni"       # Moon in Leo
    assert c["planets"]["Saturn"]["sign"] == "Aquarius"
    assert c["planets"]["Saturn"]["dignity"] == "own"           # Aquarius is Saturn's
    assert c["planets"]["Rahu"]["retrograde"] is True
    # Ascendant is opposite the Sun near sunset (Sun in Sagittarius -> Asc Gemini).
    assert c["ascendant"]["sign"] == "Gemini"


# -- navamsa closed form vs the movable/fixed/dual construction --------------

@requires_skyfield
def test_navamsa_matches_traditional_rule():
    from astro_engine.vedic import varga

    def traditional_d9(lon):
        sign = int(lon // 30) % 12
        part = int((lon % 30) // (10 / 3))  # 0..8
        modality = sign % 3  # 0 movable,1 fixed,2 dual
        start = {0: sign, 1: (sign + 8) % 12, 2: (sign + 4) % 12}[modality]
        return (start + part) % 12

    lons = np.linspace(0, 360, 721, endpoint=False) + 0.01
    got = varga.varga_sign(lons, 9)
    exp = np.array([traditional_d9(x) for x in lons])
    assert np.array_equal(got, exp)


@requires_skyfield
def test_varga_signs_in_range():
    from astro_engine.vedic import varga
    lons = np.linspace(0, 360, 361, endpoint=False) + 7.3
    for n in (1, 2, 3, 9, 10, 12, 30, 60):
        v = varga.varga_sign(lons, n)
        assert v.min() >= 0 and v.max() <= 11


# -- dignity rules -----------------------------------------------------------

@requires_skyfield
def test_dignity_states():
    from astro_engine.vedic import dignity
    # Sun exalted in Aries(0), debilitated in Libra(6), own in Leo(4).
    assert dignity._DIGNITY_STATES[dignity.dignity_state("Sun", np.array([0]))[0]] == "exalted"
    assert dignity._DIGNITY_STATES[dignity.dignity_state("Sun", np.array([6]))[0]] == "debilitated"
    assert dignity._DIGNITY_STATES[dignity.dignity_state("Sun", np.array([4]))[0]] == "own"
    # Saturn own in Capricorn(9)/Aquarius(10), exalted in Libra(6).
    assert dignity._DIGNITY_STATES[dignity.dignity_state("Saturn", np.array([10]))[0]] == "own"
    assert dignity._DIGNITY_STATES[dignity.dignity_state("Saturn", np.array([6]))[0]] == "exalted"


# -- houses ------------------------------------------------------------------

@requires_skyfield
def test_whole_sign_houses(vf):
    c = vf.chart(WHEN, *DELHI)
    asc_sign = c["ascendant"]["sign_index"]
    for g, p in c["planets"].items():
        expected = (p["sign_index"] - asc_sign) % 12 + 1
        assert p["house"] == expected


# -- declination out-of-bounds ----------------------------------------------

@requires_skyfield
def test_out_of_bounds_flag(sampler):
    from astro_engine.vedic import declination
    # A month of daily Moon samples must contain both OOB and non-OOB days
    # only near a standstill; here just assert the flag is consistent with dec.
    times = [datetime(2006, 1, 1, tzinfo=timezone.utc).replace(day=d) for d in range(1, 28)]
    s = sampler.sample(times, *DELHI)
    fs = declination.features(s)
    oob = fs.flag["Moon_out_of_bounds"]
    assert np.array_equal(oob, np.abs(s.dec_deg["Moon"]) > s.obliquity)


# -- full battery integrity --------------------------------------------------

@requires_skyfield
def test_compute_produces_valid_featureset(vf):
    rng = np.random.default_rng(1)
    n = 400
    base = np.datetime64("1990-01-01")
    times = base + rng.integers(0, 12000, n).astype("timedelta64[D]")
    lat = rng.uniform(-55, 55, n)
    lon = rng.uniform(-180, 180, n)
    fs = vf.compute(times, lat, lon)

    assert len(fs.cat) + len(fs.flag) > 300
    # every categorical index is either -1 (N/A) or a valid category
    for name, idx in fs.cat.items():
        meta = fs.cat_meta[name]
        assert idx.min() >= -1 and idx.max() <= meta.n - 1
    # daytime should be ~half the (uniform-in-time, global) sample
    day, tot = fs.flag_count("is_daytime")
    assert 0.4 < day / tot < 0.6


@requires_skyfield
def test_module_selection(de_kernel):
    from astro_engine.vedic import VedicFeatures
    vf = VedicFeatures(de_kernel, include=["panchanga", "declination"])
    fs = vf.compute([WHEN], *DELHI)
    assert set(fs.families) <= {"panchanga", "declination"}
    with pytest.raises(ValueError):
        VedicFeatures(de_kernel, include=["nonexistent"])
