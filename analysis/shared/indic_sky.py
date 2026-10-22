"""Indian-method sky for the experiment scripts (pre-registration: docs/audit/preregistrations/preregistration_indic_sky.md).

Activated by SKY_ENGINE=ts|drk|ss (see the end of myeph.py). It replaces myeph's sky functions with
indic-ganita-engine values. Engine longitudes are sidereal in the engine's own frame; "tropical" values are
engine sidereal + myeph's Lahiri ayanamsa, so trop - ayan returns the engine's sidereal longitude exactly.
"""
import os, sys, numpy as np
_ENG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "indic-ganita-engine"))
sys.path.insert(0, _ENG)
import indic_engine
from indic_engine.compute import positions as _TS, drk as _DRK, ss as _SS

PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
_FUN = {"ts": lambda j: _TS.planet_positions(j, "TS", "madhava")["true"],
        "drk": lambda j: _DRK.planet_positions(j)["true"],
        "ss": lambda j: _SS.planet_positions(j)["true"]}

def install(g, engine):
    engine = engine.lower()
    if engine not in _FUN: raise ValueError(f"SKY_ENGINE must be modern, ts, drk or ss, not {engine!r}")
    import types
    g["_MODERN"] = types.SimpleNamespace(**{k: g[k] for k in ["sun_trop","moon_trop","geo_trop","all_trop","node_trop","speed","ayan","jd_of"]})
    pos = _FUN[engine]; ayan = g["ayan"]; cache = {}
    def sky(jd):                                   # dict of sidereal longitudes for one JD (cached)
        k = round(float(jd), 7)
        if k not in cache:
            if len(cache) > 400000: cache.clear()
            cache[k] = pos(float(jd))
        return cache[k]
    def col(name, jd):
        a = np.atleast_1d(np.asarray(jd, float)); out = np.array([sky(j)[name] for j in a])
        return out if np.ndim(jd) else float(out[0])
    def trop(name, jd): return (col(name, jd) + ayan(jd)) % 360
    def speed(name, jd): return ((col(name, np.asarray(jd, float) + 0.5) - col(name, np.asarray(jd, float) - 0.5) + 180) % 360) - 180
    def all_trop(jd):
        out = {"Sun": trop("Sun", jd), "Moon": trop("Moon", jd)}
        for p in ["Mars", "Mercury", "Jupiter", "Venus", "Saturn"]:
            out[p] = trop(p, jd); out[p + "_retro"] = speed(p, jd) < 0
        return out
    g.update(sun_trop=lambda jd: trop("Sun", jd), moon_trop=lambda jd: trop("Moon", jd),
             geo_trop=lambda name, jd: trop(name, jd), all_trop=all_trop,
             node_trop=lambda jd: trop("MoonNode", jd), speed=speed,
             SKY_ENGINE=engine, SKY_ENGINE_VERSION=indic_engine.__version__)
    print(f"[myeph] sky = indic-ganita-engine {indic_engine.__version__}, model {engine.upper()}", file=sys.stderr)
