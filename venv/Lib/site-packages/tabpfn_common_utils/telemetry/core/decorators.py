"""Decorators for recording model calls."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging
import json
import time

from dataclasses import dataclass
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from .events import ExtensionEntryEvent, FitEvent, PredictEvent
from .service import capture_event
from tabpfn_common_utils.utils import shape_of

# Logger
logger = logging.getLogger(__name__)

# Current extension
_CONTEXT_VARS = {}


def _get_context_var(var_name: str):
    """Get the shared context variable, ensuring it's the same
    instance across all imports.

    Args:
        var_name: The name of the context variable to get.

    Returns:
        The context variable.
    """
    if var_name not in _CONTEXT_VARS:
        _CONTEXT_VARS[var_name] = contextvars.ContextVar[Optional[str]](
            var_name, default=None
        )
    return _CONTEXT_VARS[var_name]


# TODO: In code with multiple estimators in a single thread it is possible that
# we'll set this config before we read it and publish it as a metric. We're
# accepting the limitation in this edge case for now in the interest of
# expediency.
def set_model_config(
    model_path: Union[str, Path], model_version: str
) -> Optional[contextvars.Token[Optional[str]]]:
    """Set the current model path.

    Args:
        model_path: The path to the model.
        model_version: The version of the model.

    Returns:
        The context variable token, or None if setting failed.
    """
    try:
        model_path = Path(model_path).name
        token = json.dumps({"model_path": model_path, "model_version": model_version})
        tok = _get_context_var("tabpfn_model_path").set(token)
        return tok
    except Exception:
        return None


def get_model_config() -> Optional[Tuple[str, str]]:
    """Get the current model path.

    Returns:
        A tuple of model_path and model_version.
    """
    token = _get_context_var("tabpfn_model_path").get()
    if token is None:
        return None

    try:
        data = json.loads(token)
        return data["model_path"], data["model_version"]
    except Exception:
        return None


def set_init_params(
    params: Dict[str, Any],
) -> Optional[contextvars.Token[Optional[str]]]:
    """Set the initial parameters of the model.

    Args:
        params: The initial parameters of the model.
    """
    try:
        token = json.dumps(params)
        tok = _get_context_var("tabpfn_model_init_params").set(token)
        return tok
    except Exception:
        return None


def get_init_params() -> Optional[Dict[str, Any]]:
    """Get the initial parameters of the model.

    Returns:
        The initial parameters of the model.
    """
    token = _get_context_var("tabpfn_model_init_params").get()
    if token is None:
        return None

    try:
        return json.loads(token)
    except Exception:
        return None


def get_current_extension() -> Optional[str]:
    """Get the current extension.

    Returns:
        The name of the current extension.
    """
    return _get_context_var("tabpfn_current_extension").get()


@contextlib.contextmanager
def _extension_context(extension_name: str):
    """Context manager to set the current extension.

    Args:
        extension_name: The name of the extension to set.
    """
    context_var = _get_context_var("tabpfn_current_extension")
    tok = context_var.set(extension_name)
    try:
        yield
    finally:
        context_var.reset(tok)


# Marker attribute for wrapped functions
_MARKER_ATTR = "_tabpfn_extension_name"


def _already_wrapped(fn, extension_name: str) -> bool:
    """Check whether a callable is already wrapped for the given extension.

    Args:
        fn: The callable to check.
        extension_name: The extension name.

    Returns:
        True if the callable already carries the marker.
    """
    f = fn
    while True:
        if getattr(f, _MARKER_ATTR, None) == extension_name:
            return True

        inner = getattr(f, "__wrapped__", None)
        if inner is None:
            return False

        f = inner


def _is_public(name: str) -> bool:
    """Check whether an attribute name is public.

    Args:
        name: The attribute name.

    Returns:
        True if the name does not start with an underscore.
    """
    return not name.startswith("_")


def _is_sync_callable(fn) -> bool:
    """Check whether a callable is synchronous (not async/generator-async).

    Args:
        fn: The callable to check.

    Returns:
        True if the callable is synchronous.
    """
    if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        return False
    return True


def _wrap_callable_with_extension(fn, extension_name: str):
    """Wrap a synchronous callable so it runs under the extension context.

    Skips wrapping if the callable is async or already wrapped for this extension.

    Args:
        fn: The callable to wrap.
        extension_name: The extension name to set during the call.

    Returns:
        The wrapped callable, or the original if no wrapping was needed.
    """
    if not _is_sync_callable(fn) or _already_wrapped(fn, extension_name):
        return fn

    @wraps(fn)
    def wrapped(*args, **kwargs):
        # Don't override an outer context
        if _get_context_var("tabpfn_current_extension").get() is not None:
            return fn(*args, **kwargs)
        with _extension_context(extension_name):
            capture_event(ExtensionEntryEvent(extension_name=extension_name))
            return fn(*args, **kwargs)

    setattr(wrapped, _MARKER_ATTR, extension_name)
    return wrapped


def _wrap_class(cls, extension_name: str, public_only: bool = True):
    """Wrap selected methods on a class so they run under the extension context.

    Args:
        cls: The class to modify in place.
        extension_name: The extension name to set during wrapped method calls.
        public_only: Whether to wrap only public methods.

    Returns:
        The modified class.
    """
    # Determine descriptors to skip
    try:
        from functools import cached_property

        _skip_descriptors = (property, cached_property)
    except Exception:
        _skip_descriptors = (property,)

    for name, attr in list(cls.__dict__.items()):
        # Skip non-public methods
        if public_only and not _is_public(name):
            continue

        # Skip descriptors
        if isinstance(attr, _skip_descriptors):
            continue

        # Wrap static/class methods
        if isinstance(attr, (staticmethod, classmethod)):
            fn = attr.__func__
            wrapped_fn = _wrap_callable_with_extension(fn, extension_name)
            if wrapped_fn is not fn:
                setattr(cls, name, type(attr)(wrapped_fn))

        # Wrap regular functions
        elif inspect.isfunction(attr):
            wrapped_fn = _wrap_callable_with_extension(attr, extension_name)
            if wrapped_fn is not attr:
                setattr(cls, name, wrapped_fn)

    return cls


def set_extension(extension_name: str, public_only: bool = True):
    """Decorator to set the current extension.

    Args:
        extension_name: The name of the extension to set.
        public_only: Whether to wrap only public methods when decorating a class.

    Returns:
        A decorator that can be used on functions or classes.
    """

    def deco(obj):
        if inspect.isclass(obj):
            return _wrap_class(obj, extension_name, public_only=public_only)
        return _wrap_callable_with_extension(obj, extension_name)

    return deco


# Type of model tasks
ModelTaskType = Literal["classification", "regression"]
ModelMethodType = Literal["fit", "predict"]

# Event resolver
_EVENT_BY_METHOD: dict[ModelMethodType, type[FitEvent] | type[PredictEvent]] = {
    "fit": FitEvent,
    "predict": PredictEvent,
}


def track_model_call(model_method: ModelMethodType, param_names: list[str]) -> Callable:
    """Decorator that tracks model calls.

    Args:
        model_method: Model execution method, `fit` or `predict`.
        param_names: List of parameters to track for.

    Example:
        @track_model_call(model_method="fit", param_names=["X_test", ...])
        def prepare(...):
    """

    def decorator(func: Callable) -> Callable:
        # Validate parameter names at decoration time
        signature = inspect.signature(func)

        func_param_names = set(signature.parameters.keys())
        for param_name in param_names:
            if param_name not in func_param_names:
                raise ValueError(f"Parameter {param_name} not declared")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _safe_call_with_telemetry(
                func, args, kwargs, model_method, param_names
            )

        return wrapper

    return decorator


def _safe_call_with_telemetry(
    func: Callable,
    args: tuple,
    kwargs: dict,
    model_method: ModelMethodType,
    param_names: list[str],
) -> Any | None:
    """Execute function with telemetry, handling all exceptions internally.

    Args:
        func: The function to execute, decorated.
        args: Positional arguments.
        kwargs: Keyword arguments.
        model_method: Model execution method.
        param_names: List of parameters to track for.

    Returns:
        Tuple of (result, call_info).
    """
    call_info = None

    # Step 1: Pick up call information using introspection
    try:
        call_info = _make_callinfo(
            func,
            model_method,
            param_names,
            *args,
            **kwargs,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Call info failed with: {e}")

    # Step 2: Run the actual function
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = int((time.perf_counter() - start) * 1000)

    # Step 3: Send telemetry event
    if call_info is not None:
        try:
            _send_model_called_event(call_info, duration_ms)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Telemetry failed for {func.__name__}: {e}")

    return result


def _send_model_called_event(call_info: _ModelCallInfo, duration_ms: int) -> None:
    """Send telemetry event for a model call.

    Args:
        call_info: Call information.
        duration_ms: Duration in milliseconds.
    """
    # Task (regression | classification) is required
    if call_info.task is None:
        return

    # Infer event type and dimensionality
    _event_cls = _EVENT_BY_METHOD.get(call_info.model_method)
    if not _event_cls or len(call_info.shapes) < 1:
        return

    # Build event
    num_rows, num_columns = _extract_shape_info(call_info.shapes)
    event_kwargs = {
        "task": call_info.task,
        "num_rows": num_rows,
        "num_columns": num_columns,
        "duration_ms": duration_ms,
    }

    # Create event, might fail due to a type mismatch
    try:
        event = _event_cls(**event_kwargs)
        # Set the extension name or None
        event.extension = get_current_extension()

        # Set the model path or None
        config = get_model_config()
        if config is not None:
            # Unpack the model config
            model_path, model_version = config

            # Set the model path and version
            event.model_path = model_path
            event.model_version = model_version

        # Set the model init params for fit
        if isinstance(event, FitEvent):
            event.init_params = get_init_params()

    except TypeError as e:
        logger.debug(f"Event creation failed: {e}")
        return

    # Send event, catch all backend exceptions
    try:
        capture_event(event)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Event capture failed: {e}")
        return


def _extract_shape_info(shapes: dict[str, tuple[int, ...]]) -> tuple[int, int]:
    """Extract total samples and features from shapes dictionary.

    Args:
        shapes: Dictionary of parameter names to their shapes

    Returns:
        Tuple of (num_rows, num_columns)
    """
    num_rows, num_columns = 0, 0
    for shape in shapes.values():
        if len(shape) >= 2 and shape[1] > 1:  # X data: (samples, features)
            num_rows += shape[0]
            num_columns += shape[1]
        elif len(shape) == 1:  # y data: (samples,)
            num_rows += shape[0]
    return num_rows, num_columns


@dataclass
class _StackFrame:
    """A single frame in the call stack."""

    function_name: str | None
    module_name: str | None


@dataclass
class _ModelCallInfo:
    """Info about a decorated model call, its arguments and call stack."""

    shapes: dict[str, tuple[int, ...]]
    task: Literal["classification", "regression"]
    model_method: Literal["fit", "predict"]


def _capture_call_stack(func: Callable, max_frames: int = 25) -> list[_StackFrame]:
    """Capture the call stack for a model call function.

    Args:
        func: The function to capture the call stack for.
        max_frames: The maximum number of frames to capture.

    Returns:
        The call stack.
    """
    frames: list[_StackFrame] = []

    # Capture the decorated function's module and file
    func_mod = getattr(func, "__module__", None)
    frame = _StackFrame(getattr(func, "__name__", None), func_mod)
    frames.append(frame)

    f = inspect.currentframe()

    # Skip internal functions
    internal = {"_capture_call_stack", "_make_callinfo", "wrapper", "track_model_call"}
    while f and f.f_code.co_name in internal:
        f = f.f_back

    # Capture the call stack
    while f and len(frames) < max_frames:
        # Skip internal functions
        if f.f_code.co_name in internal:
            f = f.f_back
            continue

        # Get the module name
        m = inspect.getmodule(f)
        mod_name = m.__name__ if m else f.f_globals.get("__name__")

        frame = _StackFrame(f.f_code.co_name, mod_name)
        frames.append(frame)

        # Move to the next frame
        f = f.f_back

    return frames


def _make_callinfo(
    func: Callable,
    model_method: ModelMethodType,
    param_names: list[str],
    *args: Any,
    **kwargs: Any,
) -> _ModelCallInfo | None:
    """Collect model call information.

    Args:
        func: Called and decorated function.
        *args: Positional input arguments.
        **kwargs: Keyword arguments for the function.
        model_method: Model execution method.
        param_names: List of parameters to track for.
    """
    # Get the function signature
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    # Capture stack and infer the task from it
    stack = _capture_call_stack(func)

    # Infer the model task from the call stack.
    task = _infer_task(stack)
    if task is None:
        return None

    # Get the shapes based on tracked params
    shapes: dict[str, tuple[int, ...]] = {}
    for param_name in param_names:
        if param_name in bound.arguments:
            # Round the dimensionality of the dataset
            raw_shape = shape_of(bound.arguments[param_name])
            if raw_shape is not None:
                shape = _round_dims(raw_shape)
                shapes[param_name] = shape

    return _ModelCallInfo(shapes=shapes, task=task, model_method=model_method)


def _infer_task(stack: list[_StackFrame]) -> ModelTaskType | None:
    """Infer the model task from the call stack.

    Args:
        stack: The call stack.

    Returns:
        The model task.
    """
    for frame in stack:
        m = frame.module_name or ""
        if m.startswith("tabpfn.classifier"):
            return "classification"
        if m.startswith("tabpfn.regressor"):
            return "regression"
        if m.startswith("tabpfn_time_series"):
            return "regression"
    return None


def _round_dims(shape: tuple[int, int]) -> tuple[int, int]:
    """Round the dimensionality of a dataset.

    The intent is to anonymize the dataset dimensionality to prevent
    leakage of sensitive information.

    The function obscures the exact number of rows and columns in a dataset
    by rounding them up to the nearest predefined thresholds. This helps
    prevent leakage of sensitive information that might be inferred from
    precise dataset dimensions.

    Args:
        shape: The shape of the dataset.

    Returns:
        The rounded shape.
    """
    if not tuple(shape):
        return 0, 0

    # Limits for rounding the number of rows and columns
    row_limits = [10, 50, 75, 100, 150, 200, 500, 1000]

    # Limits for rounding the number of columns
    col_limits = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

    def round_dim(n: int, limits: list[int]) -> int:
        for limit in limits:
            if n <= limit:
                return limit
        return (n // 50) * 50

    num_rows = round_dim(shape[0], row_limits)
    num_columns = round_dim(shape[1], col_limits)
    return num_rows, num_columns
