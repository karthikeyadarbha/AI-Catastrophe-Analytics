"""Enumeration of the nine classical grahas (planets + lunar nodes)."""
from enum import Enum


class PlanetName(str, Enum):
    """The nine grahas used in Vedic astrology.

    Inherits from ``str`` so members compare and serialise as their string
    value (e.g. ``PlanetName.SUN == "Sun"`` is ``True`` and ``json`` friendly).
    """

    SUN = "Sun"
    MOON = "Moon"
    MARS = "Mars"
    MERCURY = "Mercury"
    JUPITER = "Jupiter"
    VENUS = "Venus"
    SATURN = "Saturn"
    RAHU = "Rahu"   # North lunar node (mean)
    KETU = "Ketu"   # South lunar node (Rahu + 180 deg)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value
