import logging
import uuid
import textwrap

from typing import Any, Dict, Optional
from .base import render_html, PromptResult, Outcome, parse_input, PromptSpec
from ...core.state import get_property, set_property


# Logger
_logger = logging.getLogger(__name__)


# Prompt template for allowing anonymous usage analytics to be linked to a pseudonymous ID
_BODY = textwrap.dedent("""\
  Help improve TabPFN by sharing anonymous usage analytics? [y/n]

  If enabled:
  • We collect usage data linked only to a random pseudonymous ID
  • This helps us understand usage patterns and improve features

  You can turn this off at any time in your local settings.

  Privacy Policy: http://priorlabs.ai/privacy_policy/
""")


def _prompt_identity(
    *,
    title: str = "📈 Share anonymous usage analytics? (Optional)",
    body: str = _BODY,
    hint: str = "Enter `y` to accept or `n` to decline.",
) -> PromptResult:
    """Blocking IPython prompt for anonymous telemetry consent.

    Args:
        title: The title of the prompt.
        body: The body of the prompt.
        hint: The hint of the prompt.

    Returns:
        A _PromptResult object.
    """
    render_html(title, body, hint)

    def _parse(raw: str) -> tuple[Optional[Outcome], Optional[Dict[str, Any]]]:
        """Parse the user input."""
        val = raw.lower().strip()

        # Only accept explicit "y" or "n"
        if val in {"y", "yes"}:
            return "accepted", {"telemetry": True}
        elif val in {"n", "no"}:
            return "declined", {"telemetry": False}
        else:
            # Invalid input and return None to trigger retry
            return None, None

    func = parse_input(
        input_prompt="Enable anonymous usage analytics? [y/n]: ",
        on_retry_message="Invalid input. Please enter `y` or `n`.",
        parser=_parse,
        max_retries=3,
    )
    return func


def _should_prompt() -> bool:
    """
    Check if the user should be prompted to subscribe to the newsletter.
    """
    user_id = get_property("user_id")
    return user_id is None


def _on_done(res: PromptResult) -> None:
    """Done callback for newsletter prompt.

    Args:
        res: The prompt result.
    """
    if res.outcome != "accepted":
        return

    user_id = str(uuid.uuid4())
    set_property("user_id", user_id)


class IdentityPrompt:
    """Prompt that is used to allow linking anonymous usage
    to a pseudonymous ID.
    """

    @staticmethod
    def build() -> PromptSpec:
        """Construct the prompt specification.

        Returns:
            The prompt specification.
        """
        prompt = PromptSpec(
            kind="identity",
            trigger=_should_prompt,
            ask=_prompt_identity,
            on_done=_on_done,
        )
        return prompt
