"""JSON configuration loader."""
import json
from pathlib import Path
from typing import Any, Optional, Union


def load_json(path: Union[str, Path], *, default: Optional[Any] = None) -> Any:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.
        default: Value returned when the file does not exist.

    Returns:
        The parsed JSON, or ``default`` if the file is missing.

    Raises:
        ValueError: If the file exists but contains invalid JSON.
    """
    p = Path(path)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse JSON file at {p}: {e}")
