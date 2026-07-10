"""Direction of apparent planetary motion."""
from enum import Enum


class MotionType(str, Enum):
    """Apparent motion of a planet as seen from Earth.

    ``str`` mixin keeps values JSON-friendly and comparable to plain strings,
    but prefer comparing against the enum members (``MotionType.RETROGRADE``)
    rather than raw strings.
    """

    DIRECT = "direct"
    RETROGRADE = "retrograde"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value
