"""The EngineManager: wires an ephemeris backend to the plugin registry."""
import importlib
import logging
from typing import Dict

from astro_engine.models.context import Context
from astro_engine.models.event_repository import EventRepository
from astro_engine.core.registry import PluginRegistry
from astro_engine.core.exceptions import EphemerisError
from astro_engine.adapters.base import EphemerisEngineBase

logger = logging.getLogger("astro_engine")

# Backends are referenced by dotted path and imported lazily so that installing
# only one backend's dependency (e.g. skyfield but not pyswisseph) still works.
_ADAPTER_PATHS: Dict[str, str] = {
    "swiss": "astro_engine.adapters.swiss.engine.SwissEphemerisEngine",
    "jpl": "astro_engine.adapters.jpl.engine.JplEphemerisEngine",
}


class EngineManager:
    """Main orchestrator: builds the ephemeris backend and runs plugins."""

    def __init__(self, adapter_name: str = "swiss", **adapter_kwargs):
        """Create the manager.

        Args:
            adapter_name: Ephemeris backend to use (``'swiss'`` or ``'jpl'``).
            **adapter_kwargs: Passed to the backend constructor (e.g.
                ``ephe_path`` for swiss, ``ephemeris``/``cache_dir`` for jpl).
        """
        adapter_cls = self._load_adapter_class(adapter_name)
        self.ephemeris_engine: EphemerisEngineBase = adapter_cls(**adapter_kwargs)
        self.plugin_registry = PluginRegistry()
        logger.info("EngineManager initialised with '%s' adapter.", self.ephemeris_engine.name)
        logger.info("Available plugins: %s", self.plugin_registry.list_available_plugins())

    @staticmethod
    def available_backends() -> list:
        """Return the names of all registered ephemeris backends."""
        return list(_ADAPTER_PATHS)

    @staticmethod
    def _load_adapter_class(adapter_name: str):
        if adapter_name not in _ADAPTER_PATHS:
            raise ValueError(
                f"Unknown adapter: {adapter_name}. "
                f"Available: {list(_ADAPTER_PATHS)}"
            )
        module_path, class_name = _ADAPTER_PATHS[adapter_name].rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise EphemerisError(
                f"Backend '{adapter_name}' is unavailable because a dependency "
                f"is missing: {e}. Install the appropriate extra "
                f"(e.g. 'pip install astro-engine[jpl]' or '[swiss]')."
            ) from e
        return getattr(module, class_name)

    def run(self, context: Context) -> EventRepository:
        """Run the configured plugins and aggregate their events.

        Plugins listed in ``context.plugin_configs`` are run (unless a value is
        ``False``, which disables that plugin). If ``plugin_configs`` is empty,
        every registered plugin runs.
        """
        master_repo = EventRepository()

        plugins_to_run = (
            list(context.plugin_configs.keys())
            if context.plugin_configs
            else self.plugin_registry.list_available_plugins()
        )

        logger.info("Executing plugins: %s", plugins_to_run)

        for plugin_name in plugins_to_run:
            if context.plugin_configs.get(plugin_name) is False:
                logger.info("Skipping disabled plugin: '%s'", plugin_name)
                continue

            logger.info("Running plugin: '%s'", plugin_name)
            plugin = self.plugin_registry.get_plugin(plugin_name)
            plugin_repo = plugin.run(context, self.ephemeris_engine)
            master_repo.extend(plugin_repo)

        logger.info("Aggregation complete. Found %d event(s).", len(master_repo))
        return master_repo
