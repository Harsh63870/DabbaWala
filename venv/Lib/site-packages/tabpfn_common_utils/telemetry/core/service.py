import logging
import os

from datetime import datetime
from posthog import Posthog
from .config import download_config
from .events import BaseTelemetryEvent
from .runtime import get_execution_context
from ...utils import singleton
from typing import Any, Dict, Optional


# Set up logging
logger = logging.getLogger(__name__)

# Suppress PostHog and analytics-python logs (unless explicitly enabled)
if os.getenv("TABPFN_ENABLE_TELEMETRY_LOGS", "0").lower() not in ("1", "true"):
    # Supress warnings
    for logger_name in ["posthog", "analytics", "posthog.analytics"]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


@singleton
class ProductTelemetry:
    """
    Service for capturing anonymous and aggregated telemetry data.
    """

    # Public PostHog project API key
    PROJECT_API_KEY = "phc_wemJSoR4e8rGJomHz0clm6aHdOWp0EvpRP368uVsUvJ"

    # Public PostHog host (EU)
    HOST = "https://eu.i.posthog.com"

    # PostHog client instance
    _posthog_client: Optional[Posthog] = None

    def __init__(
        self,
        max_queue_size: int = 10,
        flush_at: int = 10,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Telemetry service.
        """
        # If telemetry is disabled, don't initialize the PostHog client
        if not self.telemetry_enabled():
            self._posthog_client = None
            return

        if api_key is None:
            api_key = self.PROJECT_API_KEY

        # Initialize the PostHog client
        self._posthog_client = Posthog(
            project_api_key=api_key,
            host=self.HOST,
            disable_geoip=True,
            enable_exception_autocapture=False,
            max_queue_size=max_queue_size,
            flush_at=flush_at,
        )

    @classmethod
    def telemetry_enabled(cls) -> bool:
        """
        Check if telemetry is enabled.

        Returns:
            bool: True if telemetry is enabled, False otherwise.
        """
        # Disable telemetry by default in CI environments, but allow override
        exec_context = get_execution_context()
        default_disable = "1" if exec_context.ci else "0"

        disable_telemetry = os.getenv(
            "TABPFN_DISABLE_TELEMETRY", default_disable
        ).lower()
        if disable_telemetry in ("1", "true"):
            return False

        # Overwrite any settings based on server-side configuration
        config = download_config()
        if config["enabled"] is False:
            return False

        return True

    def capture(
        self,
        event: BaseTelemetryEvent,
        *,
        distinct_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Send an event. Optionally override distinct_id (useful in API).

        Args:
            event (BaseTelemetryEvent): The event to send.
            distinct_id (str): The distinct ID to send the event to.
            properties (dict): The additional properties attached to an event.
            timestamp (datetime): The timestamp of the event.
        """
        if self.telemetry_enabled() is False:
            return

        # Add the check just for type safety
        if self._posthog_client is None:
            return

        # Merge the event properties with the provided properties
        properties = {**event.properties, **(properties or {})}

        # Set up dynamic `capture` arguments
        capture_args = {
            "event": event.name,
            "properties": properties,
            "timestamp": timestamp or event.timestamp,
        }

        # Do not capture any user ID if it is not provided
        if distinct_id:
            capture_args["distinct_id"] = distinct_id

        try:
            self._posthog_client.capture(**capture_args)
        except Exception:
            # Silently ignore any errors
            pass

    def flush(self) -> None:
        """
        Flush the PostHog client telemetry queue.
        """
        if self.telemetry_enabled() is False:
            return

        try:
            self._posthog_client.flush()  # type: ignore
        except Exception:
            # Silently ignore any errors
            pass


def capture_event(
    event: BaseTelemetryEvent, max_queue_size: int = 10, flush_at: int = 10
) -> None:
    """Capture a telemetry event.

    Args:
        event: The event to capture.
        max_queue_size: The maximum size of the queue.
        flush_at: The number of events to flush.
    """
    client = ProductTelemetry(max_queue_size, flush_at)
    client.capture(event)
