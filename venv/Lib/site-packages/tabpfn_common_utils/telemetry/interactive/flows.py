"""Client code for anonymously tracking model usage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from typing import List
from ..core.config import download_config
from ..core.state import get_property, set_property
from ..core import PingEvent, SessionEvent, capture_event
from .prompts.base import PromptSpec
from .prompts.newsletter import NewsletterPrompt
from .prompts.identity import IdentityPrompt
from ..core.runtime import get_execution_context


def capture_session(enabled: bool = True) -> None:
    """Capture a session event.

    Args:
        enabled: Whether to capture a session event.
    """
    if not enabled:
        return

    event = SessionEvent()
    capture_event(event)


def ping(enabled: bool = True) -> None:
    """Ping the usage service to track usage and analytics.

    Args:
        enabled: Whether to ping the usage service.
    """
    # Skip if disabled
    if not enabled:
        return

    utc_now = datetime.now(timezone.utc)

    # Determine whether we should ping
    frequency_days = {
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
    }
    for frequency, days in frequency_days.items():
        key = f"last_pinged_at_{frequency}"
        last_pinged_at = get_property(key, data_type=datetime)

        # Maintain backward compatibility, avoid duplicates
        if frequency == "daily" and not last_pinged_at:
            last_pinged_at = get_property("last_pinged_at", data_type=datetime)

        # Check if the time since the last ping is greater than the delta days
        if last_pinged_at and utc_now - last_pinged_at < timedelta(days=days):
            continue

        # Ping the usage service
        event = PingEvent(frequency=frequency)  # type: ignore[arg-type]
        capture_event(event)

        # Acknowledge the ping
        set_property(key, utc_now)


def _trigger_prompts(delta_days: int, max_prompts: int) -> bool:
    """Determine if prompts should be shown.

    Args:
        now: The current time.
        delta_days: The number of days to wait between prompts.
        max_prompts: The maximum number of prompts to show.

    Returns:
        True if a prompt should be shown, False otherwise.
    """
    utc_now = datetime.now(timezone.utc)

    # Download prompt configuration
    config = download_config()

    # By default, don't prompt
    if not config.get("prompt_user", False):
        return False

    # If new installation, don't prompt
    install_date = get_property("install_date", data_type=datetime)
    if not install_date:
        set_property("install_date", utc_now)
        return False

    # Avoid prompt in first 24 hours
    delta_hours = config.get("prompt_delta_hours", 24)

    # Ensure timezone-aware datetime
    if install_date.tzinfo is None:
        install_date = install_date.replace(tzinfo=timezone.utc)

    if utc_now - install_date <= timedelta(hours=delta_hours):
        return False

    # If used <= 5 times, don't prompt
    nr_usages = get_property("nr_usages", 0, data_type=int)
    set_property("nr_usages", nr_usages + 1)

    if nr_usages < config.get("prompt_nr_usages", 5):
        return False

    # If last prompted > 30 days, prompt
    last_prompted_at = get_property("last_prompted_at", data_type=datetime)
    if not last_prompted_at:
        return True

    # Check if the maximum number of prompts has been reached
    nr_prompts = get_property("nr_prompts", 0, data_type=int)
    if nr_prompts >= max_prompts:
        return False

    # Check if the time since the last prompt is greater than the delta days
    if utc_now - last_prompted_at >= timedelta(days=delta_days):
        return True

    return False


def _build_prompts() -> List[PromptSpec]:
    """Build the prompts that will be run in sequence.

    Returns:
        The prompts that will be run in sequence.
    """
    classes = [NewsletterPrompt, IdentityPrompt]
    return [cls.build() for cls in classes]


def opt_in(enabled: bool = True, delta_days: int = 30, max_prompts: int = 2) -> None:
    """Run the opt-in flows in sequence (blocking, data-driven).

    Args:
        enabled: Whether to run the opt-in flows.
        delta_days: Minimum days between prompts per kind.
        max_prompts: Max number of times to show each prompt.
    """
    if not enabled:
        return

    # Only show prompts in Jupyter/IPython
    exec_context = get_execution_context()
    if exec_context.kernel not in {"jupyter", "ipython"}:
        return

    # Check if prompts should be shown
    if not _trigger_prompts(delta_days, max_prompts):
        return

    # Build the required prompts
    prompts = _build_prompts()

    # Run each needed prompt in order
    for spec in prompts:
        # Skip if value already present
        if not spec.trigger():
            continue

        # Ask (blocking, Jupyter/IPython-safe) and handle result
        result = spec.ask()
        spec.on_done(result)

    # Acknowledge the prompt
    set_property("last_prompted_at", datetime.now(timezone.utc))
    nr_prompts = get_property("nr_prompts", 0, data_type=int)
    set_property("nr_prompts", nr_prompts + 1)
