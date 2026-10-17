import logging
import os
import re

import requests
import textwrap
from typing import Any, Dict, Optional
from .base import render_html, PromptResult, Outcome, parse_input, PromptSpec
from ...core.state import get_property, set_property


# Logger
_logger = logging.getLogger(__name__)

# Constants
_API_URL = os.environ.get(
    "TABPFN_API_URL", "https://tabpfn-server-wjedmz7r5a-ez.a.run.app"
)

# Prompt template for subscribing to newsletter
_BODY = textwrap.dedent("""\
  Would you like to receive occasional email updates about TabPFN?

  These may include:
  • New features
  • Bug fixes
  • Research highlights

  ➡ If yes, please enter your email address.
  ➡ If not, just press Enter to skip.

  Your email will only be used to send you these updates.
  You can unsubscribe at any time via the link in each email.
  We will not share your email with third parties.

  Privacy Policy: http://priorlabs.ai/privacy_policy/
""")


def _prompt_newsletter(
    *,
    title: str = "📬 Subscribe to TabPFN updates? (Optional)",
    body: str = _BODY,
    hint: str = "Enter your email to opt in, or press Enter to skip.",
) -> PromptResult:
    """Blocking IPython prompt for newsletter subscription.

    Args:
        title: The title of the prompt.
        body: The body of the prompt.
        hint: The hint of the prompt.

    Returns:
        A _PromptResult object.
    """
    # Render the HTML
    render_html(title, body, hint)

    def _parser(raw: str) -> tuple[Outcome, Optional[Dict[str, Any]]]:
        """Parse the user input."""
        if _is_valid_email(raw):
            return "accepted", {"email": raw}

        return "declined", {}

    # Parse the user input
    func = parse_input(
        input_prompt="Email (optional, press Enter to skip): ",
        parser=_parser,
    )
    return func


def _is_valid_email(text: str) -> bool:
    """Check if text looks like a valid email address.

    Args:
        text: The text to check.

    Returns:
        True if the text looks like an email address, False otherwise.
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, text))


def _subscribe_user(email: str) -> None:
    """Subscribe the user to newsletter using the API.

    Args:
        email: The email of the user to opt_in.
    """
    endpoint = _API_URL + "/newsletter/subscribe"
    r = requests.post(endpoint, json={"email": email.lower()}, timeout=5)
    if r.status_code != 200:
        _logger.debug(f"Failed to subscribe user {email}: {r.text}")
        return


def _should_prompt() -> bool:
    """
    Check if the user should be prompted to subscribe to the newsletter.
    """
    email = get_property("email")
    return email is None


def _on_done(res: PromptResult) -> None:
    """Done callback for newsletter prompt.

    Args:
        res: The prompt result.
    """
    if res.outcome != "accepted":
        return

    payload = res.data or {}
    email = payload.get("email")
    if not email:
        return

    try:
        # Subscribe the user to the newsletter using the API
        _subscribe_user(email)
    except Exception:
        # best-effort; don't disrupt host process
        pass

    set_property("email", email)


class NewsletterPrompt:
    """Prompt that is used to subscribe the user to the newsletter."""

    @staticmethod
    def build() -> PromptSpec:
        """Construct the prompt specification.

        Returns:
            The prompt specification.
        """
        prompt = PromptSpec(
            kind="newsletter",
            trigger=_should_prompt,
            ask=_prompt_newsletter,
            on_done=_on_done,
        )
        return prompt
