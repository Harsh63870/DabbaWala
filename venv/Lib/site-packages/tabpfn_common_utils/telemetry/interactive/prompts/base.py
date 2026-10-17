import html

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

# Outcome of a prompt
Outcome = Literal["accepted", "declined", "dismissed", "timeout"]

# Jupyter notebook template HTML code
_JUPYTER_HTML = """
<div style="
    border:1px solid #e5e7eb;border-radius:10px;padding:12px;margin:6px 0;
    font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, sans-serif;">
  <div style="font-weight:600;margin-bottom:6px;">{title}</div>
  <div style="white-space:pre-line;margin-bottom:6px;">{body}</div>
  <div style="white-space:pre-line;margin-bottom:8px;color:#4b5563;">{hint}</div>
  <div style="margin-top:8px;">
    <a href="https://priorlabs.ai/privacy_policy/" target="_blank"
        style="color:#3b82f6;text-decoration:none;font-size:14px;"
    >Privacy Policy</a>
  </div>
</div>
"""


@dataclass(frozen=True)
class PromptResult:
    """Result of a prompt"""

    outcome: Outcome
    data: Dict[str, Any]


@dataclass(frozen=True)
class PromptSpec:
    """Specification of a prompt"""

    kind: Literal["newsletter", "identity"]
    trigger: Callable[[], bool]
    ask: Callable[[], PromptResult]
    on_done: Callable[[PromptResult], None]


def render_html(title: str, body: str, hint: str) -> None:
    """Render a rich HTML block in IPython.

    Args:
        title: The title of the prompt.
        body: The body of the prompt.
        hint: The hint of the prompt.
    """
    from IPython.display import HTML, display  # type: ignore

    html_block = _JUPYTER_HTML.format(
        title=html.escape(title),
        body=html.escape(body),
        hint=html.escape(hint),
    )
    display(HTML(html_block))


def parse_input(
    input_prompt: str,
    parser: Callable[[str], tuple[Optional[Outcome], Optional[Dict[str, Any]]]],
    max_retries: int = 3,
    on_retry_message: str = "Invalid input. Please try again.",
) -> PromptResult:
    """Blocking input loop with unified parsing and retry logic.

    Args:
        input_prompt: The input() prompt string.
        parser: A callable that receives the raw string and parses outcome.
        max_retries: Maximum number of retries for invalid input.

    Returns:
        A PromptResult object.
    """
    retries = 0

    while retries <= max_retries:
        try:
            raw = input(input_prompt).strip()
        except KeyboardInterrupt:
            return PromptResult("dismissed", {})

        outcome, data = parser(raw)

        # If outcome is None, input was invalid - retry
        if outcome is None:
            retries += 1
            if retries <= max_retries:
                continue
            else:
                # Max retries exceeded, treat as dismissed
                return PromptResult("dismissed", {})

        # Valid outcome, return result
        return PromptResult(outcome, data or {})

    return PromptResult("dismissed", {})
