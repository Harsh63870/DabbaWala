"""Interactive telemetry containing prompts and runtime detection."""

from __future__ import annotations


try:
    # Import the specific functions to expose
    from .flows import ping, opt_in, capture_session

    __all__ = ["ping", "opt_in", "capture_session"]

except ImportError:

    def __getattr__(name):
        e = ImportError(
            "Interactive telemetry is not available. "
            "Install with: pip install tabpfn_common_utils[interactive]"
        )
        raise e

    __all__ = []
