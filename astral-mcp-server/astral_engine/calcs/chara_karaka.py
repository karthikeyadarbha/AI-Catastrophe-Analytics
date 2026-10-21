"""
Jaimini Chara Karaka calculation and transition detection engine.
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional
from ..core.models import Planet, PlanetPosition, DateRange, TransitionEvent
from ..constants import CHARA_GRAHAS, CHARA_KARAKA_CODES, CHARA_KARAKA_NAMES

def compute_chara_karakas(positions: Dict[Planet, PlanetPosition], use_8_karakas: bool = False) -> Dict[str, str]:
    """
    Ranks eligible planets by descending order of degree within current sign.
    Returns mapping: {"AK": "Moon", "AmK": "Mars", ...}
    """
    eligible = [positions[Planet(name)] for name in CHARA_GRAHAS if Planet(name) in positions]
    
    # Sort descending by degree_in_sign
    sorted_planets = sorted(eligible, key=lambda p: p.degree_in_sign, reverse=True)
    
    mapping = {}
    for code, p in zip(CHARA_KARAKA_CODES, sorted_planets):
        mapping[code] = p.planet.value
    return mapping

def enrich_with_chara_karakas(positions: Dict[Planet, PlanetPosition], use_8_karakas: bool = False):
    """Mutates PlanetPosition objects in-place with their assigned Chara Karaka."""
    mapping = compute_chara_karakas(positions, use_8_karakas)
    # Reverse mapping: Planet -> Code
    planet_to_code = {p_name: code for code, p_name in mapping.items()}
    for planet, pos in positions.items():
        if planet.value in planet_to_code:
            pos.chara_karaka = planet_to_code[planet.value]
        else:
            pos.chara_karaka = None
