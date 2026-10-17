from __future__ import annotations

from .core.events import DatasetEvent, FitEvent, ModelLoadEvent, PingEvent, PredictEvent
from .core.service import ProductTelemetry, capture_event
from .core.decorators import (
    track_model_call,
    set_extension,
    get_current_extension,
    set_model_config,
    set_init_params,
    get_init_params,
)

# Public exports
__all__ = [
    "DatasetEvent",
    "ModelLoadEvent",
    "FitEvent",
    "PingEvent",
    "PredictEvent",
    "ProductTelemetry",
    "capture_event",
    "track_model_call",
    "set_extension",
    "get_current_extension",
    "set_model_config",
    "set_init_params",
    "get_init_params",
]
