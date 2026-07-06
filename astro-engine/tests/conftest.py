"""Shared pytest fixtures and capability detection.

The JPL tests need a DE kernel. Resolution order:

1. ``$ASTRO_TEST_DE_KERNEL`` -- path to an existing ``.bsp`` file.
2. A ``de421.bsp`` already cached under ``tests/.cache``.
3. Download ``de421.bsp`` into ``tests/.cache`` (needs network; ~17 MB).

If none of these succeed the JPL fixtures ``skip`` rather than fail, so the
suite stays green on machines without the optional dependency or network.
"""
import importlib.util
import os
from pathlib import Path

import pytest

CACHE_DIR = Path(__file__).parent / ".cache"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_SWISS = _module_available("swisseph")
HAS_SKYFIELD = _module_available("skyfield")

requires_swiss = pytest.mark.skipif(not HAS_SWISS, reason="pyswisseph not installed")
requires_skyfield = pytest.mark.skipif(not HAS_SKYFIELD, reason="skyfield not installed")


@pytest.fixture(scope="session")
def de_kernel() -> str:
    """Return a path to a usable DE kernel, or skip if unavailable."""
    if not HAS_SKYFIELD:
        pytest.skip("skyfield not installed")

    env = os.environ.get("ASTRO_TEST_DE_KERNEL")
    if env and Path(env).is_file():
        return env

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / "de421.bsp"
    if cached.is_file():
        return str(cached)

    # Last resort: download (marked network-dependent).
    try:
        from skyfield.api import Loader

        loader = Loader(str(CACHE_DIR))
        loader.load("de421.bsp")
    except Exception as exc:  # pragma: no cover - network/offline guard
        pytest.skip(f"could not obtain a DE kernel: {exc}")
    return str(cached)


@pytest.fixture(scope="session")
def swiss_engine():
    if not HAS_SWISS:
        pytest.skip("pyswisseph not installed")
    from astro_engine import AstroEngine

    return AstroEngine(backend="swiss")


@pytest.fixture(scope="session")
def jpl_engine(de_kernel):
    from astro_engine import AstroEngine

    return AstroEngine(backend="jpl", ephemeris=de_kernel)
