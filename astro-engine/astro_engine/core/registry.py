"""Discovers and lazily instantiates plugins from a JSON registry."""
import importlib
from pathlib import Path
from typing import Dict, List, Type, Optional

from astro_engine.core.interface import PluginInterface
from astro_engine.core.exceptions import PluginError
from astro_engine.utils.config_loader import load_json

_DEFAULT_REGISTRY = Path(__file__).parent.parent / "config" / "plugin_registry.json"


class PluginRegistry:
    """Discovers, lazy-loads and provides access to event-detection plugins.

    Reads a JSON file mapping plugin names to dotted class paths and
    instantiates each class only when first requested.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = _DEFAULT_REGISTRY

        config = load_json(registry_path)
        if config is None:
            raise PluginError(f"Plugin registry not found at {registry_path}.")
        self._registry_config: Dict[str, str] = config
        self._loaded_plugins: Dict[str, PluginInterface] = {}

    def get_plugin(self, name: str) -> PluginInterface:
        """Return the plugin instance registered under ``name`` (cached).

        Raises:
            PluginError: If the plugin is not registered or fails to load.
        """
        if name in self._loaded_plugins:
            return self._loaded_plugins[name]

        if name not in self._registry_config:
            raise PluginError(f"Plugin '{name}' not found in registry.", plugin_name=name)

        class_path = self._registry_config[name]
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            plugin_class: Type[PluginInterface] = getattr(module, class_name)
            plugin_instance = plugin_class()
        except (ImportError, AttributeError, ValueError) as e:
            raise PluginError(
                f"Failed to load plugin '{name}' from path '{class_path}': {e}",
                plugin_name=name,
            ) from e

        if plugin_instance.name != name:
            raise PluginError(
                f"Plugin name mismatch. Expected '{name}', but plugin at "
                f"'{class_path}' is named '{plugin_instance.name}'.",
                plugin_name=name,
            )

        self._loaded_plugins[name] = plugin_instance
        return plugin_instance

    def list_available_plugins(self) -> List[str]:
        """Return the names of all registered plugins."""
        return list(self._registry_config.keys())
