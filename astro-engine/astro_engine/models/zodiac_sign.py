"""The 12 rasis (zodiac signs), named in the South-Indian Sanskrit tradition."""
from enum import Enum


class ZodiacSign(str, Enum):
    """The 12 rasis in zodiacal order (Mesham = Aries ... Meenam = Pisces)."""

    MESHAM = "Mesham"          # Aries
    VRISHABAM = "Vrishabam"    # Taurus
    MITHUNAM = "Mithunam"      # Gemini
    KATAKAM = "Katakam"        # Cancer
    SIMHAM = "Simham"          # Leo
    KANYA = "Kanya"            # Virgo
    TULA = "Tula"              # Libra
    VRISCHIKAM = "Vrischikam"  # Scorpio
    DHANASSU = "Dhanassu"      # Sagittarius
    MAKARAM = "Makaram"        # Capricorn
    KUMBHAM = "Kumbham"        # Aquarius
    MEENAM = "Meenam"          # Pisces

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value
