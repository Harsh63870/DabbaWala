"""Headless telemetry API containing event schemas and capture."""

from __future__ import annotations

from .events import (
    BaseTelemetryEvent,
    ExtensionEntryEvent,
    ModelLoadEvent,
    PingEvent,
    DatasetEvent,
    FitEvent,
    PredictEvent,
    SessionEvent,
)
from .service import ProductTelemetry, capture_event
from .decorators import (
    track_model_call,
    set_extension,
    get_current_extension,
    set_model_config,
    set_init_params,
    get_init_params,
)

# Public exports
__all__ = [
    "BaseTelemetryEvent",
    "ExtensionEntryEvent",
    "PingEvent",
    "DatasetEvent",
    "ModelLoadEvent",
    "FitEvent",
    "PredictEvent",
    "ProductTelemetry",
    "SessionEvent",
    "capture_event",
    "track_model_call",
    "set_extension",
    "get_current_extension",
    "set_model_config",
    "set_init_params",
    "get_init_params",
]
