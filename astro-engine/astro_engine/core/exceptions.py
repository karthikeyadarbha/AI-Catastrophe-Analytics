"""Exception hierarchy for the Astro Engine."""


class AstroError(Exception):
    """Base class for all Astro Engine errors."""

    def __init__(self, message: str, *, code: str = None):
        super().__init__(message)
        self.code = code


class EphemerisError(AstroError):
    """Raised when an ephemeris backend fails to compute a position."""

    def __init__(self, message: str, *, code: str = "EPEM01"):
        super().__init__(message, code=code)


class PluginError(AstroError):
    """Raised when a plugin cannot be loaded or fails while running."""

    def __init__(self, message: str, *, plugin_name: str = None, code: str = "PLGN01"):
        full_msg = f"[{plugin_name}] {message}" if plugin_name else message
        super().__init__(full_msg, code=code)
        self.plugin_name = plugin_name
