"""
app/tools.py
Dispatcher tools for DabbaETA Agent:
1. predict_eta(order)
2. list_risky_orders(threshold)
3. draft_message(order, language)
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import sentry_sdk

from app.predictor import get_predictor
from app.schemas import OrderSchema

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dabbaeta.db")
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "orders.csv")

def get_orders_df(limit: int = 50) -> pd.DataFrame:
    """Reads orders from SQLite database or fallback CSV."""
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(f"SELECT * FROM orders ORDER BY rowid DESC LIMIT {limit}", conn)
            conn.close()
            if not df.empty:
                return df
        except Exception:
            pass

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        return df.head(limit)

    # Empty fallback
    return pd.DataFrame(columns=[
        "Order_ID", "Customer_Name", "Customer_Phone", "Delivery_Location",
        "Distance_km", "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type",
        "Preparation_Time_min", "Courier_Experience_yrs", "promised_time_min"
    ])


def tool_predict_eta(order: Dict[str, Any], model_name: str = "tabpfn") -> Dict[str, Any]:
    """
    Predicts the expected delivery time (ETA) and probability of delay for an order.
    """
    predictor = get_predictor(model_name)
    df = pd.DataFrame([order])

    # Ensure required default columns
    defaults = {
        "Order_ID": "ORD-LIVE",
        "Customer_Name": "Customer",
        "Customer_Phone": "+91 98000 00000",
        "Delivery_Location": "Local Destination",
        "Distance_km": 4.5,
        "Weather": "Clear",
        "Traffic_Level": "Medium",
        "Time_of_Day": "Evening",
        "Vehicle_Type": "Scooter",
        "Preparation_Time_min": 15.0,
        "Courier_Experience_yrs": 2.0,
        "promised_time_min": 40.0
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    eta = float(predictor.predict(df)[0])
    promised = float(df["promised_time_min"].iloc[0])
    late_prob = float(predictor.predict_late_probability(df, np.array([promised]))[0]) if hasattr(predictor, "predict_late_probability") else (0.8 if eta > promised else 0.2)

    return {
        "order_id": str(df["Order_ID"].iloc[0]),
        "customer_name": str(df["Customer_Name"].iloc[0]),
        "eta_minutes": round(eta, 1),
        "promised_time_min": round(promised, 1),
        "delay_minutes": max(0.0, round(eta - promised, 1)),
        "late_risk_probability": round(late_prob, 3),
        "is_late_predicted": bool(eta > promised or late_prob >= 0.5),
        "model_used": predictor.name,
    }


def tool_list_risky_orders(threshold: float = 0.45, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Scans today's pending orders and identifies orders at high risk of breaching promised delivery time.
    """
    df = get_orders_df(limit=limit)
    if df.empty:
        return []

    predictor = get_predictor("tabpfn")
    etas = predictor.predict(df)
    promised_times = df["promised_time_min"].fillna(40.0).values
    probs = predictor.predict_late_probability(df, promised_times)

    risky = []
    for idx, row in df.iterrows():
        eta = float(etas[idx])
        promised = float(promised_times[idx])
        prob = float(probs[idx])
        if prob >= threshold or eta > promised:
            risky.append({
                "order_id": row.get("Order_ID", f"DB-{10000+idx}"),
                "customer_name": row.get("Customer_Name", "Customer"),
                "customer_phone": row.get("Customer_Phone", "+91 98000 00000"),
                "delivery_location": row.get("Delivery_Location", "City Center"),
                "distance_km": float(row.get("Distance_km", 3.0)),
                "traffic_level": row.get("Traffic_Level", "Medium"),
                "weather": row.get("Weather", "Clear"),
                "eta_minutes": round(eta, 1),
                "promised_time_min": round(promised, 1),
                "expected_delay_min": max(0.0, round(eta - promised, 1)),
                "late_risk_probability": round(prob, 2),
                "risk_badge": "HIGH" if prob > 0.7 else "MODERATE"
            })

    # Sort descending by risk probability
    risky.sort(key=lambda x: x["late_risk_probability"], reverse=True)
    return risky


def tool_draft_message(order: Dict[str, Any], language: str = "Hinglish") -> Dict[str, Any]:
    """
    Generates a polite, empathetic WhatsApp delay alert for tiffin / bakery customers.
    Languages: Hinglish, Hindi, English.
    """
    cust_name = order.get("customer_name") or order.get("Customer_Name") or "Customer"
    order_id = order.get("order_id") or order.get("Order_ID") or "Order"
    delay_min = order.get("expected_delay_min") or order.get("delay_minutes") or 15
    delay_rounded = int(round(float(delay_min)))
    if delay_rounded < 5:
        delay_rounded = 15

    traffic = order.get("traffic_level") or order.get("Traffic_Level") or "heavy traffic"
    weather = order.get("weather") or order.get("Weather") or "Clear"

    lang_lower = language.lower()

    if "hind" in lang_lower and "ing" not in lang_lower:
        # Pure Hindi
        text = (
            f"नमस्ते {cust_name} जी! 🙏\n"
            f"आपके स्वादिष्ट भोजन का आर्डर ({order_id}) रास्ते में है। रास्ते में {traffic} और मौसम के कारण डिलीवरी में लगभग {delay_rounded} मिनट की देरी हो सकती है।\n"
            f"हमारा डिलीवरी पार्टनर पूरी सावधानी से आप तक पहुँच रहा है। असुविधा के लिए हमें अत्यंत खेद है!\n"
            f"- डब्बा ईटीए (DabbaETA Tiffin Express)"
        )
    elif "eng" in lang_lower:
        # English
        text = (
            f"Hello {cust_name}! 👋\n"
            f"Your order ({order_id}) is currently out for delivery. Due to {traffic} and current weather conditions, we anticipate a slight delay of ~{delay_rounded} minutes.\n"
            f"Our rider is prioritizing your safety and warm meal arrival. Thank you for your patience and support!\n"
            f"- DabbaETA Dispatch Team"
        )
    else:
        # Default Hinglish (warm, authentic Indian small business voice)
        text = (
            f"Namaste {cust_name} ji! 🙏\n"
            f"Aapka dabba order ({order_id}) dispatch ho chuka hai. Raste mein {traffic} ki wajah se lagbhag {delay_rounded} minutes ka delay ho sakta hai.\n"
            f"Humare delivery bhaiya full speed aur safety ke saath nikal chuke hain. Inconvenience ke liye dil se maafi! ❤️\n"
            f"- Aapka DabbaETA Delivery Partner"
        )

    return {
        "order_id": order_id,
        "customer_name": cust_name,
        "language": language,
        "draft_text": text
    }

# Tool Registry for agent reflection
AVAILABLE_TOOLS = {
    "predict_eta": tool_predict_eta,
    "list_risky_orders": tool_list_risky_orders,
    "draft_message": tool_draft_message,
}
