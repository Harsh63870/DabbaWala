"""State management for telemetry."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from platformdirs import user_config_dir
from typing import Any, Optional


# Check if filelock is available
# ruff: noqa: I001
_HAS_FILELOCK = False
try:
    import filelock  # type: ignore[import-untyped] # noqa: F401

    _HAS_FILELOCK = True
except ImportError:
    pass

# Application name and vendor
APP = ".tabpfn"
VENDOR = "priorlabs"
FILENAME = "state.json"

# Schema
_DEFAULT_STATE: dict[str, Any] = {
    # Date and time when state file was created
    "created_at": None,
    # Anonymous user ID, only with consent
    "user_id": None,
    # Email address, only with consent
    "email": None,
    # Number of times prompts were shown
    "nr_prompts": 0,
    # Date and time when last prompt was shown
    "last_prompted_at": None,
    # Date and time when last anonymous ping was sent
    "last_pinged_at": None,
}


def _safe_state_path() -> Optional[Path]:
    """Get the path to the state file safely.

    If an exception is raised in the underlying _state_path() function,
    return None.

    Returns:
        Optional[Path]: The path to the state file or None.
    """
    try:
        return _state_path()
    except Exception:
        return None


def _state_path() -> Path:
    """Get the path to the state file.

    Returns:
        Path: The path to the state file.
    """
    # Overrides first
    if p := os.getenv("TABPFN_STATE_PATH"):
        return Path(p).expanduser()
    if d := os.getenv("TABPFN_STATE_DIR"):
        return Path(d).expanduser() / FILENAME

    # Standard per-user config dir
    return Path(user_config_dir(APP, VENDOR)) / FILENAME


def _ensure_dir(path: Path) -> None:
    """Ensure the directory exists.

    Args:
        path: The path to the directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.parent.chmod(0o700)


def _write_with_lock(path: Path, data: dict[str, Any]) -> None:
    """Write data to file with filelock if available.

    Args:
        path: The path to the state file.
        data: The data to write to the state file.
    """
    if _HAS_FILELOCK:
        _write_with_filelock(path, data)
    else:
        _write_without_lock(path, data)


def _write_with_filelock(path: Path, data: dict[str, Any]) -> None:
    """Write data to file using filelock for thread safety.

    Args:
        path: The path to the state file.
        data: The data to write to the state file.
    """
    import filelock  # type: ignore[import-untyped]

    lock_path = path.with_suffix(".lock")
    lock = filelock.FileLock(lock_path, timeout=10)

    with lock:
        _safe_write_data_to_file(path, data)


def _write_without_lock(path: Path, data: dict[str, Any]) -> None:
    """Write data to file without filelock (fallback).

    Args:
        path: The path to the state file.
        data: The data to write to the state file.
    """
    _safe_write_data_to_file(path, data)


def _safe_write_data_to_file(path: Path, data: dict[str, Any]) -> None:
    """Write data to file atomically safely.

    Args:
        path: The path to the state file.
        data: The data to write to the state file.
    """
    try:
        _write_data_to_file(path, data)
    except Exception:
        pass


def _write_data_to_file(path: Path, data: dict[str, Any]) -> None:
    """Write data to file atomically.

    Args:
        path: The path to the state file.
        data: The data to write to the state file.
    """
    # Ensure the directory exists
    _ensure_dir(path)

    # Create a temporary file
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            _write_json_to_file(f, data)
            _sync_file_to_disk(f)

        # Atomic replace
        Path(tmp).replace(path)
        _set_file_permissions(path)
    finally:
        _cleanup_temp_file(tmp)


def _write_json_to_file(file, data: dict[str, Any]) -> None:
    """Write JSON data to file.

    Args:
        file: The file object to write to.
        data: The data to write.
    """
    params = {
        "ensure_ascii": False,
        "separators": (",", ":"),
        "default": _json_serialize,
    }
    json.dump(data, file, **params)


def _sync_file_to_disk(file) -> None:
    """Sync file to disk.

    Args:
        file: The file object to sync.
    """
    file.flush()
    os.fsync(file.fileno())


def _json_serialize(obj: Any) -> Any:
    """Default JSON encoder that handles datetime objects.

    Args:
        obj: The object to potentially encode.

    Returns:
        The object with datetime objects converted to ISO strings.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()

    return obj


def _set_file_permissions(path: Path) -> None:
    """Set file permissions to 0o600, ignoring any errors.

    Args:
        path: The path to the file.
    """
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def _cleanup_temp_file(tmp_path: str) -> None:
    """Remove temporary file, ignoring any errors.

    Args:
        tmp_path: The path to the temporary file.
    """
    with contextlib.suppress(OSError):
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def _read(path: Path) -> dict[str, Any]:
    """Read the state file.

    Args:
        path: The path to the state file.

    Returns:
        The state data (empty dict on errors).
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        # Corrupt -> treat as empty
        return {}


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw state to the current schema.

    Returns:
        A merged state dict conforming to _DEFAULT_STATE.
    """
    if not raw:
        state = dict(_DEFAULT_STATE)
        state["created_at"] = datetime.now(timezone.utc).isoformat()
        return state

    # Backfill defaults without dropping unknown future keys
    return {**_DEFAULT_STATE, **raw}


def load_state() -> dict[str, Any]:
    """Load the telemetry state.

    Returns:
        The telemetry state (schema-normalized).
    """
    path = _safe_state_path()
    if path is None:
        return _normalize({})

    raw = _read(path)
    return _normalize(raw)


def save_state(state: dict[str, Any]) -> None:
    """Save the telemetry state.

    Args:
        state: The telemetry state to persist.
    """
    normalized = _normalize(state)
    if not normalized.get("created_at"):
        normalized["created_at"] = datetime.now(timezone.utc).isoformat()

    path = _safe_state_path()
    if path is None:
        return

    try:
        # Safely write, silently ignore any errors
        _write_with_lock(path, normalized)
    except Exception:
        pass


def get_property(key: str, default: Any = None, data_type: type | None = None) -> Any:
    """Get a property from the telemetry state with optional type conversion.

    Args:
        key: The property name.
        default: The default value if the property is not found.
        data_type: The expected data type.

    Returns:
        The property value or default.
    """
    state = load_state()
    value = state.get(key, default)

    if value is None or value == default:
        return value

    if data_type is None:
        return value

    # TODO: handle other type conversions
    try:
        if data_type == datetime:
            # datetime conversion from ISO format string
            if isinstance(value, str):
                value = datetime.fromisoformat(value)
        else:
            # Handle other type conversions
            value = data_type(value)
    except (ValueError, TypeError):
        return default
    else:
        return value


def set_property(key: str, value: Any) -> Any:
    """Set a property on the telemetry state.

    Args:
        key: The property name.
        value: The value to set.

    Returns:
        The value that was set.
    """
    state = load_state()
    state[key] = value
    save_state(state)
    return value
