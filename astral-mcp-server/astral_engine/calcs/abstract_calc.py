"""
Abstract ephemeris calculation interface.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from ..core.models import Planet, PlanetPosition

class AbstractCalc(ABC):
    @abstractmethod
    def get_planet_position(self, planet: Planet, dt: datetime, lat: float = 0.0, lon: float = 0.0, elev: float = 0.0) -> PlanetPosition:
        pass

    @abstractmethod
    def get_all_positions(self, dt: datetime, lat: float = 0.0, lon: float = 0.0, elev: float = 0.0) -> dict[Planet, PlanetPosition]:
        pass
