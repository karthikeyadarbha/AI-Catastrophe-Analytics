"""
Dynamic Celestial Feature Generator.
Compiles and evaluates customizable feature sets for arbitrary dates and observer locations.
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from ..core.models import Planet
from ..calcs.sidereal_calc import SiderealCalc
from ..calcs.chara_karaka import compute_chara_karakas
from ..algorithms.aspect_engine import AspectEngine
from ..utils import minimal_angular_diff

PRESET_CONFIGS = {
    "minimal": {
        "planets": ["all"],
        "attributes": ["longitude"],
        "include_chara_karakas": False,
        "include_stelliums": False,
    },
    "ml_cyclical": {
        "planets": ["all"],
        "attributes": ["longitude", "speed", "is_retrograde", "sin_cos", "elongation"],
        "include_chara_karakas": False,
        "include_stelliums": True,
    },
    "astrological": {
        "planets": ["all"],
        "attributes": ["sign", "degree_in_sign", "nakshatra", "pada", "is_retrograde", "is_combust"],
        "include_chara_karakas": True,
        "include_stelliums": True,
    },
    "seismic_tidal": {
        "planets": ["Sun", "Moon", "Mars", "Jupiter", "Saturn"],
        "attributes": ["longitude", "altitude", "elongation", "sin_cos"],
        "pairwise_angles": [["Moon", "Sun"], ["Jupiter", "Saturn"], ["Mars", "Sun"]],
        "include_chara_karakas": False,
        "include_stelliums": True,
    },
    "financial_astro": {
        "planets": ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Rahu"],
        "attributes": ["longitude", "speed", "is_retrograde", "is_combust", "sin_cos", "sign"],
        "pairwise_angles": [["Mercury", "Sun"], ["Venus", "Sun"], ["Jupiter", "Saturn"], ["Mars", "Saturn"]],
        "include_chara_karakas": True,
        "include_stelliums": True,
    },
    "exhaustive": {
        "planets": ["all"],
        "attributes": [
            "longitude", "latitude", "speed", "sign", "degree_in_sign",
            "nakshatra", "pada", "is_retrograde", "is_combust", "sin_cos", "altitude", "elongation"
        ],
        "pairwise_angles": [
            ["Moon", "Sun"], ["Mercury", "Sun"], ["Venus", "Sun"],
            ["Mars", "Sun"], ["Jupiter", "Sun"], ["Saturn", "Sun"],
            ["Jupiter", "Saturn"], ["Mars", "Saturn"], ["Mars", "Jupiter"]
        ],
        "include_chara_karakas": True,
        "include_stelliums": True,
    },
}

class FeatureSpec(BaseModel):
    """Declarative specification of celestial features to populate."""
    preset: Optional[str] = Field(None, description="Preset name: minimal, ml_cyclical, astrological, seismic_tidal, financial_astro, exhaustive")
    planets: Optional[List[str]] = Field(default_factory=lambda: ["all"], description="List of planet names or ['all']")
    attributes: Optional[List[str]] = Field(default_factory=lambda: ["longitude", "speed", "sign", "is_retrograde"], description="Attributes per planet")
    pairwise_angles: Optional[List[List[str]]] = Field(default_factory=list, description="Pairs of planets for angular separation and aspect analysis, e.g. [['Moon', 'Sun'], ['Jupiter', 'Saturn']]")
    include_chara_karakas: bool = Field(False, description="Whether to include Jaimini Chara Karakas")
    include_stelliums: bool = Field(False, description="Whether to compute stellium clusters in zodiac signs")
    include_ayanamsa: bool = Field(False, description="Whether to include ayanamsa offset value")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, preset: Optional[str] = None) -> FeatureSpec:
        """Create a FeatureSpec resolving presets and overrides."""
        if not data:
            data = {}
        target_preset = preset or data.get("preset")
        base = {}
        if target_preset and target_preset in PRESET_CONFIGS:
            base = dict(PRESET_CONFIGS[target_preset])
            base["preset"] = target_preset

        # Merge explicit user overrides on top of base preset
        for k, v in data.items():
            if v is not None:
                base[k] = v

        return cls(**base)


class FeatureGenerator:
    """Computes dynamic, on-demand feature dictionaries for timestamps."""
    def __init__(self, calc: SiderealCalc, spec: FeatureSpec):
        self.calc = calc
        self.spec = spec
        self.aspect_engine = AspectEngine(calc)

        # Resolve target planets
        all_planet_enums = [
            Planet.SUN, Planet.MOON, Planet.MARS, Planet.MERCURY,
            Planet.JUPITER, Planet.VENUS, Planet.SATURN, Planet.RAHU, Planet.KETU
        ]
        if not self.spec.planets or "all" in [p.lower() for p in self.spec.planets]:
            self.target_planets = all_planet_enums
        else:
            self.target_planets = [Planet.from_string(p) for p in self.spec.planets]

    def generate(self, dt: datetime, lat: float = 0.0, lon: float = 0.0, elev_m: float = 0.0) -> Dict[str, Any]:
        """Generate the requested feature dictionary for a given datetime."""
        features: Dict[str, Any] = {}

        # 1. Compute positions
        has_coords = (lat != 0.0 or lon != 0.0)
        positions = self.calc.get_all_positions(dt, lat=lat if has_coords else 0.0, lon=lon if has_coords else 0.0, elev=elev_m)
        sun_pos = positions.get(Planet.SUN)

        attrs = [a.lower() for a in (self.spec.attributes or [])]

        # 2. Extract requested attributes for target planets
        for p in self.target_planets:
            pos = positions.get(p)
            if not pos:
                continue
            p_name = p.value

            if "longitude" in attrs or "lon" in attrs:
                features[f"{p_name}_Lon"] = round(pos.longitude, 4)
            if "latitude" in attrs or "lat" in attrs:
                features[f"{p_name}_Lat"] = round(pos.latitude, 4)
            if "speed" in attrs:
                features[f"{p_name}_Speed"] = round(pos.speed, 4)
            if "sign" in attrs:
                features[f"{p_name}_Sign"] = pos.sign
            if "sign_index" in attrs:
                features[f"{p_name}_Sign_Idx"] = pos.sign_index
            if "degree_in_sign" in attrs or "deg_in_sign" in attrs:
                features[f"{p_name}_Deg_In_Sign"] = round(pos.degree_in_sign, 4)
            if "nakshatra" in attrs:
                features[f"{p_name}_Nakshatra"] = pos.nakshatra
            if "nakshatra_index" in attrs:
                features[f"{p_name}_Nakshatra_Idx"] = pos.nakshatra_index
            if "pada" in attrs:
                features[f"{p_name}_Pada"] = pos.pada
            if "is_retrograde" in attrs or "retro" in attrs:
                features[f"{p_name}_Is_Retro"] = (pos.motion.value == "Retrograde")
            if "is_combust" in attrs or "combust" in attrs:
                features[f"{p_name}_Is_Combust"] = pos.is_combust

            # Machine Learning Cyclical Encodings (sine and cosine of angular coordinate)
            if "sin_cos" in attrs or "cyclical" in attrs:
                rad = math.radians(pos.longitude)
                features[f"{p_name}_Sin_Lon"] = round(math.sin(rad), 6)
                features[f"{p_name}_Cos_Lon"] = round(math.cos(rad), 6)

            # Solar Elongation
            if "elongation" in attrs and sun_pos and p != Planet.SUN:
                features[f"{p_name}_Elongation"] = round(minimal_angular_diff(pos.longitude, sun_pos.longitude), 4)

            # Topocentric Altitude / Azimuth
            if ("altitude" in attrs or "alt" in attrs) and pos.altitude_deg is not None:
                features[f"{p_name}_Altitude"] = round(pos.altitude_deg, 2)

        # 3. Pairwise relationships
        for pair in (self.spec.pairwise_angles or []):
            if len(pair) == 2:
                b1_str, b2_str = pair[0], pair[1]
                pair_key = f"{b1_str}_{b2_str}"
                try:
                    rel = self.aspect_engine.get_angular_relationship(b1_str, b2_str, dt)
                    features[f"{pair_key}_Sep_Deg"] = rel["angular_separation_deg"]
                    features[f"{pair_key}_Aspect"] = rel["aspect"] or "None"
                    features[f"{pair_key}_Is_Aligned"] = rel["is_aligned"]
                    if rel["illumination_pct"] is not None:
                        features[f"{pair_key}_Illumination"] = rel["illumination_pct"]
                except Exception:
                    pass

        # 4. Jaimini Chara Karakas
        if self.spec.include_chara_karakas:
            karakas = compute_chara_karakas(positions)
            features["Atmakaraka"] = karakas.get("AK")
            features["Amatyakaraka"] = karakas.get("AmK")
            features["Bhratrukaraka"] = karakas.get("BK")
            features["Matrukaraka"] = karakas.get("MK")
            features["Putrakaraka"] = karakas.get("PK")
            features["Gnatikaraka"] = karakas.get("GK")
            features["Darakaraka"] = karakas.get("DK")

        # 5. Stelliums
        if self.spec.include_stelliums:
            sign_counts: Dict[str, int] = {}
            for p, pos in positions.items():
                if p != Planet.KETU:  # Ketu is opposite Rahu
                    sign_counts[pos.sign] = sign_counts.get(pos.sign, 0) + 1
            max_cluster = max(sign_counts.values()) if sign_counts else 0
            features["Max_Stellium_Count"] = max_cluster
            features["Has_Stellium_4Plus"] = (max_cluster >= 4)

        # 6. Ayanamsa
        if self.spec.include_ayanamsa:
            features["Ayanamsa_Name"] = self.calc.ayanamsa_name

        return features
