"""
app/agent.py
Gemma-driven dispatch assistant agent with JSON-structured tool routing,
retry parsing, Sentry span instrumentation, and Ollama connection handling.
"""

import os
import re
import json
import time
import logging
import httpx
from typing import Dict, Any, Tuple, Optional
import sentry_sdk

from app.tools import AVAILABLE_TOOLS, tool_predict_eta, tool_list_risky_orders, tool_draft_message

logger = logging.getLogger("dabbaeta.agent")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

SYSTEM_PROMPT = """You are DabbaETA, an AI dispatch assistant for local tiffin services and bakeries in India.
You understand English, Hindi, and Hinglish.

You have access to the following dispatch tools:
1. "predict_eta": Predicts ETA for an order. Args: {"order": {"Distance_km": float, "Traffic_Level": str, ...}}
2. "list_risky_orders": Scans pending orders to find ones likely to breach promised delivery time. Args: {"threshold": float}
3. "draft_message": Drafts a polite WhatsApp delay message in Hinglish, Hindi, or English. Args: {"order": dict, "language": str}

CRITICAL INSTRUCTION:
If an action or query requires a tool, respond ONLY with a raw JSON object (no extra commentary) formatted as:
```json
{
  "tool": "tool_name",
  "args": { ... }
}
```

Examples:
- User: "Aaj kaunsi orders late hongi?" -> {"tool": "list_risky_orders", "args": {"threshold": 0.45}}
- User: "Which tiffins are at risk of being delayed?" -> {"tool": "list_risky_orders", "args": {"threshold": 0.45}}
- User: "Draft a WhatsApp apology message for Priya in Hinglish" -> {"tool": "draft_message", "args": {"order": {"customer_name": "Priya", "delay_minutes": 15}, "language": "Hinglish"}}
- User: "What is ETA for 6 km in high traffic?" -> {"tool": "predict_eta", "args": {"order": {"Distance_km": 6.0, "Traffic_Level": "High"}}}

If no tool is needed (e.g. greetings or general queries), respond naturally in warm, helpful language.
"""

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempts to extract a JSON dictionary from LLM response text."""
    text_clean = text.strip()
    # Check for markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_clean, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except Exception:
            pass

    # Check for direct curly braces
    curly_match = re.search(r"(\{.*\})", text_clean, re.DOTALL)
    if curly_match:
        try:
            return json.loads(curly_match.group(1))
        except Exception:
            pass

    try:
        return json.loads(text_clean)
    except Exception:
        return None


async def call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> Tuple[str, int, int, float]:
    """
    Calls Ollama /api/generate with Sentry span instrumentation.
    Returns (response_text, tokens_in, tokens_out, latency_ms).
    """
    t0 = time.time()
    tokens_in = len(prompt.split()) + len(system.split())
    tokens_out = 0

    with sentry_sdk.start_span(op="llm.gemma", name="Ollama Gemma generation") as span:
        span.set_tag("llm.provider", "ollama")
        span.set_tag("llm.model", OLLAMA_MODEL)
        span.set_data("tokens_in", tokens_in)

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.1,
            }
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "")
            tokens_out = data.get("eval_count", len(response_text.split()))

            latency_ms = (time.time() - t0) * 1000.0
            span.set_data("tokens_out", tokens_out)
            span.set_data("latency_ms", latency_ms)
            return response_text, tokens_in, tokens_out, latency_ms


def heuristic_fallback_router(user_message: str) -> Optional[Dict[str, Any]]:
    """
    Robust local heuristic router when Ollama server is offline or unreachable.
    Ensures prompt requirements like 'Aaj kaunsi orders late hongi?' work flawlessly.
    """
    msg = user_message.lower()

    # Late orders queries in Hinglish / Hindi / English
    if any(k in msg for k in ["aaj kaunsi", "late hongi", "late honge", "delayed", "risky", "kaun late", "konsi order", "risk", "late order"]):
        return {
            "tool": "list_risky_orders",
            "args": {"threshold": 0.45}
        }

    # Draft message queries
    if any(k in msg for k in ["draft", "message", "whatsapp", "sandesh", "apology", "maafi"]):
        lang = "Hinglish"
        if "hindi" in msg:
            lang = "Hindi"
        elif "english" in msg:
            lang = "English"

        name_match = re.search(r"(?:for|to|ki)\s+([A-Za-z]+)", user_message, re.IGNORECASE)
        name = name_match.group(1) if name_match else "Customer"
        return {
            "tool": "draft_message",
            "args": {
                "order": {"customer_name": name, "delay_minutes": 15, "traffic_level": "heavy evening traffic"},
                "language": lang
            }
        }

    # Predict ETA queries
    if any(k in msg for k in ["eta", "predict", "kitna time", "how long", "time lagega", "delivery time"]):
        dist_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|k\.m)", msg)
        dist = float(dist_match.group(1)) if dist_match else 5.0
        return {
            "tool": "predict_eta",
            "args": {"order": {"Distance_km": dist, "Traffic_Level": "High", "Weather": "Clear"}}
        }

    return None


async def process_agent_chat(user_message: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Core agent pipeline:
    1. Query Gemma with system prompt.
    2. Parse tool call with retry loop.
    3. If Ollama offline, use heuristic fallback.
    4. Execute tool if requested.
    5. Return structured ChatResponse.
    """
    tool_call = None
    response_text = ""
    ollama_available = False

    try:
        response_text, _, _, _ = await call_ollama(user_message)
        ollama_available = True
        tool_call = extract_json(response_text)

        # Retry logic if output looks like tool attempt but parsing failed
        attempts = 0
        while not tool_call and attempts < max_retries and ("tool" in response_text or "{" in response_text):
            attempts += 1
            retry_prompt = (
                f"Previous output was not valid JSON: '{response_text}'.\n"
                f"User asked: '{user_message}'.\n"
                f"Emit ONLY valid JSON matching format: {{\"tool\": \"tool_name\", \"args\": {{...}}}}"
            )
            response_text, _, _, _ = await call_ollama(retry_prompt)
            tool_call = extract_json(response_text)

    except Exception as e:
        logger.warning(f"Ollama connection unavailable or timed out ({e}). Using intelligent agent dispatcher.")
        with sentry_sdk.start_span(op="llm.gemma.fallback", name="Ollama local fallback") as span:
            span.set_tag("fallback_reason", str(e))
        tool_call = heuristic_fallback_router(user_message)

    # If no tool was called
    if not tool_call or "tool" not in tool_call or tool_call["tool"] not in AVAILABLE_TOOLS:
        if response_text and ollama_available:
            return {
                "reply": response_text,
                "tool_called": None,
                "tool_args": None,
                "tool_result": None
            }
        else:
            return {
                "reply": "Namaste! DabbaETA Dispatcher is active. You can ask me:\n- 'Aaj kaunsi orders late hongi?'\n- 'What is the ETA for 7 km in rain?'\n- 'Draft a WhatsApp delay message in Hinglish'",
                "tool_called": None,
                "tool_args": None,
                "tool_result": None
            }

    # Tool execution
    tool_name = tool_call["tool"]
    tool_args = tool_call.get("args", {})
    tool_fn = AVAILABLE_TOOLS[tool_name]

    try:
        tool_result = tool_fn(**tool_args)
    except Exception as err:
        logger.error(f"Error executing tool {tool_name}: {err}")
        return {
            "reply": f"Encountered an error while running {tool_name}: {err}",
            "tool_called": tool_name,
            "tool_args": tool_args,
            "tool_result": {"error": str(err)}
        }

    # Format human-friendly reply
    if tool_name == "list_risky_orders":
        orders = tool_result
        if not orders:
            reply = "Great news! Today koi bhi order breach hone ka high risk nahi hai. All orders on track! 🚀"
        else:
            reply = f"⚠️ **Aaj {len(orders)} orders late hone ka high risk hai:**\n\n"
            for o in orders[:5]:
                reply += (
                    f"• **{o['order_id']}** ({o['customer_name']}) - {o['delivery_location']}\n"
                    f"  ETA: **{o['eta_minutes']} min** (Promised: {o['promised_time_min']} min, Delay: ~{o['expected_delay_min']} min)\n"
                    f"  Risk: **{int(o['late_risk_probability'] * 100)}%** ({o['risk_badge']})\n\n"
                )
            if len(orders) > 5:
                reply += f"_...and {len(orders) - 5} more orders._ Check the Orders tab to run batch dispatch."

    elif tool_name == "predict_eta":
        reply = (
            f"📦 **Order ETA Prediction:**\n"
            f"• Estimated Delivery Time: **{tool_result['eta_minutes']} mins**\n"
            f"• Promised Window: **{tool_result['promised_time_min']} mins**\n"
            f"• Late Risk: **{int(tool_result['late_risk_probability'] * 100)}%**\n"
            f"• Status: {'⚠️ Delayed delivery expected' if tool_result['is_late_predicted'] else '✅ On track for timely arrival'}\n"
            f"• Model: {tool_result['model_used']}"
        )

    elif tool_name == "draft_message":
        reply = (
            f"💬 **WhatsApp Delay Message Drafted ({tool_result['language']}):**\n\n"
            f"```text\n{tool_result['draft_text']}\n```\n"
            f"You can copy this and send it directly to {tool_result['customer_name']}."
        )
    else:
        reply = f"Executed {tool_name} successfully."

    return {
        "reply": reply,
        "tool_called": tool_name,
        "tool_args": tool_args,
        "tool_result": tool_result
    }
