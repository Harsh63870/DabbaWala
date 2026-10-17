import importlib
import os
import platform
import sys
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from functools import lru_cache
from typing import Any, Dict, Literal, Optional
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlShutdown,
)
from .runtime import get_execution_context
from .state import get_property, set_property


def _uuid4() -> str:
    """
    Generate a random UUID4.

    Returns:
        str: UUID4 as a string.
    """
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    """
    Get the current UTC date and time.

    Returns:
        datetime: Current UTC date and time.
    """
    return datetime.now(timezone.utc)


def _utc_now_date() -> str:
    """Get the current UTC date as a string.

    Returns:
        str: Current UTC date in the format "YYYY-MM-DD".
    """
    return _utc_now().strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def _get_py_version() -> str:
    """
    Get the Python version as a string.

    Returns:
        str: Python version (e.g., "3.9")
    """
    version_info = sys.version_info
    return f"{version_info.major}.{version_info.minor}"


@lru_cache(maxsize=1)
def _get_sdk_version() -> str:
    """
    Get the version of the tabpfn package if it's installed.

    Returns:
        str: Version string if tabpfn is installed.
    """
    return _get_package_version("tabpfn")


@lru_cache(maxsize=1)
def _get_torch_version() -> str:
    """Get the version of the PyTorch library if it's installed.

    Returns:
        str: Version string if PyTorch is installed.
    """
    return _get_package_version("torch")


@lru_cache(maxsize=1)
def _get_sklearn_version() -> str:
    """Get the version of the scikit-learn library if it's installed.

    Returns:
        str: Version string if scikit-learn is installed.
    """
    return _get_package_version("scikit-learn", "sklearn")


@lru_cache(maxsize=1)
def _get_numpy_version() -> str:
    """Get the version of the NumPy library if it's installed.

    Returns:
        str: Version string if NumPy is installed.
    """
    return _get_package_version("numpy")


@lru_cache(maxsize=1)
def _get_pandas_version() -> str:
    """Get the version of the Pandas library if it's installed.

    Returns:
        str: Version string if Pandas is installed.
    """
    return _get_package_version("pandas")


@lru_cache(maxsize=1)
def _get_autogluon_version() -> str:
    """Get the version of the AutoGluon library if it's installed.

    Returns:
        str: Version string if AutoGluon is installed.
    """
    return _get_package_version("autogluon.core")


def _get_gpu_type() -> Optional[str]:
    """Detect a local GPU using PyTorch (the TabPFN dependency) and return its
    human-readable name.

    Returns:
        Optional[str]: Human-readable name of the GPU if available.
    """
    # First, we try to use the NVML library to get the GPU names
    # This is the preferred method as it is faster than using PyTorch
    nvml_initialized = False

    try:
        nvmlInit()

        # Set a flag to indicate that the NVML library was initialized
        # if running on CPU, this part of the code will not be reached
        nvml_initialized = True

        counts = nvmlDeviceGetCount()
        if counts == 0:
            return None

        # Retrieve the names of the devices
        devices: list[str] = [
            nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)) for i in range(counts)
        ]

        # Because NVML runs very fast, we just return the device name
        # without caching it. We return the first device name and assume
        # that the VM has the same GPU type for all devices.
        return devices[0]
    except Exception:
        pass
    finally:
        # Shutdown the NVML library
        if nvml_initialized:
            nvmlShutdown()

    # Only then, as an alternative, we try to use PyTorch to get the GPU name
    # This is the slowest method as it requires importing the PyTorch library
    # so we do not prefer this over the previous methods
    return _get_torch_gpu_type()


@lru_cache(maxsize=1)
def _get_torch_gpu_type() -> Optional[str]:
    """Get the type of GPU using PyTorch.

    Returns:
        Optional[str]: Type of GPU if available.
    """
    # First, we try to load the torch module from the sys.modules
    # because we can assume it is already loaded and we avoid
    # re-initializing it.
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch  # type: ignore[import]
        except ImportError:
            return None

    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) or "unknown"
            return name

    except Exception:  # noqa: BLE001
        pass

    # Apple Silicon via MPS
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # torch doesn't expose an MPS "model string"
            name = "Apple M-series GPU (MPS)"
            return name
    except Exception:
        pass

    return None


def _get_package_version(dist_name: str, module_name: Optional[str] = None) -> str:
    """Read the package distribution metadata and return the version.

    Args:
        dist_name: Name of the package to read the metadata for.
        module_name: Name of the module to read the version for, if different
            from the distribution name.

    Returns:
        str: Version string if the package is installed, None otherwise.
    """
    try:
        return version(dist_name)
    except PackageNotFoundError:
        # Fallback to importing the package and getting the version
        try:
            module = importlib.import_module(module_name or dist_name)
            return getattr(module, "__version__", "unknown")
        except ImportError:
            return "unknown"


@lru_cache(maxsize=1)
def _get_platform_os() -> str:
    """Get the operating system of the platform.

    Returns:
        str: Operating system of the platform.
    """
    return platform.system()


@lru_cache(maxsize=1)
def _get_runtime_kernel() -> Optional[str]:
    """Get the runtime environment of the platform.

    Returns:
        str: Runtime environment of the platform.
    """
    exec_context = get_execution_context()
    return exec_context.kernel


@lru_cache(maxsize=1)
def _get_runtime_environment() -> Optional[str]:
    """Get the runtime environment of the platform.

    Returns:
        str: Runtime environment of the platform.
    """
    exec_context = get_execution_context()
    return exec_context.environment


@lru_cache(maxsize=1)
def _get_install_id() -> str:
    """Get or create the install ID. If not set in disk-cached
    state, generate a new one and store it in the state.

    Returns:
        str: The install ID.
    """
    install_id = get_property("install_id")

    # Fallback to new install ID
    if install_id is None:
        install_id = _uuid4()
        set_property("install_id", install_id)

    return install_id


@lru_cache(maxsize=1)
def _get_install_date() -> str:
    """Return the install date as YYYY-MM-DD.

    In case the user was using TabPFN before this change was made,
    the install date will not correspond to the actual install date,
    but rather when the user upgraded the page and made a first call.

    Returns:
        str: The install date as YYYY-MM-DD.
    """
    value = get_property("install_date")

    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except ValueError:
            pass

    # Fallback to current date
    value = _utc_now_date()
    set_property("install_date", value)
    return value


@dataclass
class BaseTelemetryEvent:
    """
    Base class for all telemetry events.
    """

    # Python version that the SDK is running on
    python_version: str = field(default_factory=_get_py_version, init=False)

    # TabPFN version that the SDK is running on
    tabpfn_version: str = field(default_factory=_get_sdk_version, init=False)

    # Timestamp of the event
    timestamp: datetime = field(default_factory=_utc_now, init=False)

    # Name of the TabPFN extension making the call
    extension: Optional[str] = field(default=None, init=False)

    # Runtime environment of the platform
    runtime_kernel: Optional[str] = field(
        default_factory=_get_runtime_kernel, init=False
    )

    # Runtime environment of the platform
    runtime_environment: Optional[str] = field(
        default_factory=_get_runtime_environment, init=False
    )

    # Operating system of the platform
    platform_os: str = field(default_factory=_get_platform_os, init=False)

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def source(self) -> str:
        return os.environ.get("TABPFN_TELEMETRY_SOURCE", "sdk")

    @property
    def properties(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source
        d.pop("timestamp", None)
        d.pop("name", None)
        return d


@dataclass
class SessionEvent(BaseTelemetryEvent):
    """
    Event emitted when a session is started. A session in our case
    is a single init call to TabPFNClassifier or TabPFNRegressor.
    """

    # Install ID of the user
    install_id: str = field(default_factory=_get_install_id, init=False)

    # Install date of the user
    install_date: str = field(default_factory=_get_install_date, init=False)

    @property
    def name(self) -> str:
        return "session"


@dataclass
class ModelLoadEvent(BaseTelemetryEvent):
    """
    Event emitted when a model is loaded.
    """

    # Status of the model download attempt
    status: Literal["success", "failed"]

    # Install ID of the user
    install_id: str = field(default_factory=_get_install_id, init=False)

    # Name of the model, may be a HuggingFace repo ID
    model_name: Optional[str] = field(default=None)

    # Failure reason if the model download failed
    failure_reason: Optional[str] = field(default=None)

    def __post_init__(self):
        """Post-init hook to ensure data integrity."""
        if self.status == "success":
            self.failure_reason = None

    @property
    def name(self) -> str:
        return "model_load"


@dataclass
class PingEvent(BaseTelemetryEvent):
    """
    Event emitted when a ping is sent.
    """

    frequency: Literal["daily", "weekly", "monthly"] = "daily"

    @property
    def name(self) -> str:
        return "ping"


@dataclass
class DatasetEvent(BaseTelemetryEvent):
    """
    Event emitted when a dataset is loaded. No data is sent with this event.
    """

    # Task associated with the dataset
    task: Literal["classification", "regression"]

    # Role of the dataset in the training/testing process
    role: Literal["train", "test"]

    # Number of rows in the dataset
    num_rows: int = 0

    # Number of columns in the dataset
    num_columns: int = 0

    @property
    def name(self) -> str:
        return "dataset"


@dataclass
class ModelCallEvent(BaseTelemetryEvent):
    """
    Base class for events emitted when a model method is called (fit or predict).
    """

    # Task associated with the model call
    task: Literal["classification", "regression"]

    # Install ID of the user
    install_id: str = field(default_factory=_get_install_id, init=False)

    # Version of the PyTorch
    torch_version: str = field(default_factory=_get_torch_version, init=False)

    # Version of the scikit-learn
    sklearn_version: str = field(default_factory=_get_sklearn_version, init=False)

    # Version of the NumPy
    numpy_version: str = field(default_factory=_get_numpy_version, init=False)

    # Version of the Pandas
    pandas_version: str = field(default_factory=_get_pandas_version, init=False)

    # Version of the AutoGluon
    autogluon_version: str = field(default_factory=_get_autogluon_version, init=False)

    # Type of GPU if available
    gpu_type: Optional[str] = field(default_factory=_get_gpu_type, init=False)

    # Version of the model
    model_version: Optional[str] = field(default=None, init=False)

    # Path to the model
    model_path: Optional[str] = field(default=None, init=False)

    # Number of rows in the dataset
    num_rows: int = 0

    # Number of columns in the dataset
    num_columns: int = 0

    # Duration of the model call in milliseconds
    duration_ms: int = -1


@dataclass
class FitEvent(ModelCallEvent):
    """
    Event emitted when a model is fit.
    """

    # Initial parameters of the model
    init_params: Optional[Dict[str, Any]] = field(default=None, init=False)

    @property
    def name(self) -> str:
        return "fit_called"


@dataclass
class PredictEvent(ModelCallEvent):
    """
    Event emitted when a model is used to make predictions.
    """

    @property
    def name(self) -> str:
        return "predict_called"


@dataclass
class ExtensionEntryEvent(BaseTelemetryEvent):
    """
    Event emitted once per user-facing extension entry point call.

    Unlike FitEvent/PredictEvent which fire per downstream model call,
    this fires exactly once when the outermost @set_extension decorator
    is entered, giving an unbiased count of extension usage.
    """

    # Name of the extension that was entered
    extension_name: str = ""

    @property
    def name(self) -> str:
        return "extension_entry"
