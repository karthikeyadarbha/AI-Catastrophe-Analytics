"""The operational context passed to plugins: where, when, and what to run."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from .date_range import DateRange
from .location import Location
from .planet import PlanetName


@dataclass(frozen=True)
class Context:
    """Bundles the location, date range, planet subset and plugin config.

    Attributes:
        location: Observer location.
        date_range: Inclusive scan window.
        planets: Optional subset of planets; ``None`` means "all nine".
        plugin_configs: Maps plugin name -> config. ``False`` disables a
            plugin; any other truthy value enables it. Empty means "run all".
        metadata: Free-form user metadata carried through unchanged.
    """

    location: Location
    date_range: DateRange
    planets: Optional[List[PlanetName]] = None
    plugin_configs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.plugin_configs, dict):
            raise TypeError("plugin_configs must be a dictionary")

    def plugin_enabled(self, name: str) -> bool:
        """Check whether a specific plugin is enabled in the config."""
        config = self.plugin_configs.get(name)
        return config is not None and config is not False

    def get_plugin_config(self, name: str) -> Optional[Any]:
        """Get plugin config (can be a bool, dict, or anything)."""
        return self.plugin_configs.get(name)

    def with_planets(self, planets: List[PlanetName]) -> "Context":
        """Return a new Context restricted to the given planets."""
        return Context(
            location=self.location,
            date_range=self.date_range,
            planets=planets,
            plugin_configs=self.plugin_configs,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        lat = self.location.latitude
        lon = self.location.longitude
        return (
            f"Context(location=({lat:.2f}, {lon:.2f}), "
            f"date_range={self.date_range}, "
            f"planets={self.planets or 'all'}, "
            f"plugins={list(self.plugin_configs.keys())})"
        )
