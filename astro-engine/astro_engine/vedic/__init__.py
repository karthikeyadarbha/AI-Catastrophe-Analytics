"""Vectorized Vedic (jyotish) feature sub-libraries built on a shared sky sample.

This is the *bulk feature-extraction* face of ``astro-engine`` (as opposed to
the scalar event-detection :class:`~astro_engine.AstroEngine`). Each use case is
its own module -- :mod:`panchanga`, :mod:`bhava`, :mod:`dignity`,
:mod:`aspects`, :mod:`varga`, :mod:`dasha`, :mod:`declination`, :mod:`upagraha`,
:mod:`cycles`, :mod:`stars` -- and every one is a pure function of a
:class:`~astro_engine.vedic.sky.SkySample`.

The one-stop entry point is :class:`~astro_engine.vedic.features.VedicFeatures`.
"""
from .sky import (
    SkySample, SkySampler, GRAHAS, PHYSICAL_GRAHAS, STAR_PLANETS, NODES,
    OUTER_PLANETS, sep_deg, wrap180, wrap360,
)
from .featureset import FeatureSet, CatMeta
from .features import VedicFeatures, MODULES
from . import (
    tables, signs, panchanga, bhava, varga, dasha, dignity, aspects,
    declination, cycles, stars, upagraha,
)

__all__ = [
    # substrate
    "SkySample", "SkySampler", "GRAHAS", "PHYSICAL_GRAHAS", "STAR_PLANETS",
    "NODES", "OUTER_PLANETS", "sep_deg", "wrap180", "wrap360",
    # containers + facade
    "FeatureSet", "CatMeta", "VedicFeatures", "MODULES",
    # sub-libraries
    "tables", "signs", "panchanga", "bhava", "varga", "dasha", "dignity",
    "aspects", "declination", "cycles", "stars", "upagraha",
]
