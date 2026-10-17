"""
workflows/activities.py
Temporal activities for DabbaETA Dispatch workflow:
1. load_orders
2. predict_all (TabPFN with Sentry tracing)
3. flag_risky
4. explain (Gemma explanation of delay root-causes)
5. draft_messages (Gemma WhatsApp notifications)
"""

import os
import json
import logging
from typing import List, Dict, Any
from temporalio import activity
import sentry_sdk

from app.predictor import get_predictor
from app.tools import get_orders_df, tool_draft_message
from app.agent import call_ollama

logger = logging.getLogger("dabbaeta.workflows.activities")

@activity.defn
async def load_orders_activity(limit: int = 25) -> List[Dict[str, Any]]:
    """Loads batch of pending delivery orders."""
    df = get_orders_df(limit=limit)
    orders = df.to_dict(orient="records")
    activity.logger.info(f"Loaded {len(orders)} orders for dispatch analysis.")
    return orders

@activity.defn
async def predict_all_activity(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Predicts delivery ETA and delay risk using TabPFN."""
    import pandas as pd
    if not orders:
        return []

    df = pd.DataFrame(orders)
    predictor = get_predictor("tabpfn")

    with sentry_sdk.start_span(op="tabpfn.predict", name="Batch TabPFN Dispatch prediction") as span:
        span.set_data("batch_size", len(orders))
        span.set_tag("workflow", "DispatchWorkflow")

        etas = predictor.predict(df)
        promised_times = df["promised_time_min"].fillna(40.0).values
        probs = predictor.predict_late_probability(df, promised_times)

    results = []
    for i, order in enumerate(orders):
        eta = float(etas[i])
        promised = float(promised_times[i])
        prob = float(probs[i])
        res = dict(order)
        res["eta_minutes"] = round(eta, 1)
        res["promised_time_min"] = round(promised, 1)
        res["delay_minutes"] = max(0.0, round(eta - promised, 1))
        res["late_risk_probability"] = round(prob, 3)
        res["is_late_predicted"] = bool(eta > promised or prob >= 0.5)
        results.append(res)

    activity.logger.info(f"Completed TabPFN predictions for {len(results)} orders.")
    return results

@activity.defn
async def flag_risky_activity(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filters orders exceeding the acceptable delay risk threshold."""
    predictions = args.get("predictions", [])
    threshold = float(args.get("threshold", 0.45))

    risky = [
        p for p in predictions
        if p.get("late_risk_probability", 0.0) >= threshold or p.get("is_late_predicted", False)
    ]
    activity.logger.info(f"Identified {len(risky)} risky orders above threshold {threshold}.")
    return risky

@activity.defn
async def explain_activity(risky_orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates an executive dispatch summary explaining reasons for bottlenecks."""
    if not risky_orders:
        return {
            "summary": "All orders are running on schedule. No critical bottlenecks detected.",
            "bottlenecks": []
        }

    prompt = (
        f"You are the DabbaETA Logistics Officer. Summarize why these {len(risky_orders)} orders are facing delays:\n"
        f"{json.dumps(risky_orders[:5], indent=2)}\n"
        "Provide a concise 2-sentence operational diagnosis (e.g., peak rain, high traffic corridors, rider experience)."
    )

    explanation = ""
    try:
        reply, _, _, _ = await call_ollama(prompt)
        explanation = reply.strip()
    except Exception as e:
        logger.warning(f"Ollama unavailable for explanation ({e}). Using heuristic diagnosis.")
        high_traffic_count = sum(1 for o in risky_orders if o.get("Traffic_Level") in ("High", "Jam"))
        rain_count = sum(1 for o in risky_orders if o.get("Weather") in ("Rainy", "Stormy"))
        explanation = (
            f"Primary delay factors: {high_traffic_count} orders impacted by peak evening congestion "
            f"and {rain_count} orders navigating adverse weather conditions."
        )

    return {
        "summary": explanation,
        "risky_count": len(risky_orders)
    }

@activity.defn
async def draft_messages_activity(risky_orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drafts empathetic WhatsApp delay alerts for each customer."""
    drafts = []
    for order in risky_orders:
        msg = tool_draft_message(order, language="Hinglish")
        drafts.append({
            "order_id": order.get("Order_ID"),
            "customer_name": order.get("Customer_Name"),
            "customer_phone": order.get("Customer_Phone"),
            "delay_minutes": order.get("delay_minutes"),
            "message_text": msg.get("draft_text"),
        })
    return drafts
