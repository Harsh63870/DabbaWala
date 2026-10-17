"""
tests/test_dabbaeta.py
Pytest test suite for DabbaETA:
1. Schema validation
2. Predictor interface (TabPFN, XGBoost, LinearRegression)
3. Agent JSON tool routing & retry parser
4. Hindi/Hinglish dispatch query handling
5. WhatsApp message drafting
6. Temporal workflow activities happy path
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.schemas import OrderSchema, PredictionResult
from app.predictor import get_predictor, TabPFNPredictor
from app.baselines import XGBoostPredictor, LinearRegressionPredictor
from app.agent import extract_json, heuristic_fallback_router
from app.tools import tool_predict_eta, tool_list_risky_orders, tool_draft_message
from workflows.activities import (
    load_orders_activity,
    predict_all_activity,
    flag_risky_activity,
    explain_activity,
    draft_messages_activity
)

def test_order_schema_validation():
    """Test valid order schema creation and defaults."""
    order = OrderSchema(
        Distance_km=5.2,
        Weather="Rainy",
        Traffic_Level="High",
        Time_of_Day="Evening",
        Vehicle_Type="Motorcycle",
        Preparation_Time_min=20.0,
        Courier_Experience_yrs=3.0,
        promised_time_min=45.0
    )
    assert order.Distance_km == 5.2
    assert order.Weather == "Rainy"
    assert order.promised_time_min == 45.0
    assert order.Customer_Name == "Valued Customer"

def test_invalid_order_schema():
    """Test validation errors for invalid boundary inputs."""
    with pytest.raises(ValidationError):
        # Distance cannot be zero or negative
        OrderSchema(
            Distance_km=-2.0,
            Weather="Clear",
            Traffic_Level="Low"
        )

def test_predictor_interface():
    """Test all predictors adhere to BasePredictor contract with fit & predict."""
    df_sample = pd.DataFrame([
        {
            "Distance_km": 3.0,
            "Weather": "Clear",
            "Traffic_Level": "Low",
            "Time_of_Day": "Morning",
            "Vehicle_Type": "Scooter",
            "Preparation_Time_min": 15.0,
            "Courier_Experience_yrs": 2.0,
            "Delivery_Time_min": 25.0
        },
        {
            "Distance_km": 8.0,
            "Weather": "Rainy",
            "Traffic_Level": "High",
            "Time_of_Day": "Evening",
            "Vehicle_Type": "Motorcycle",
            "Preparation_Time_min": 22.0,
            "Courier_Experience_yrs": 1.0,
            "Delivery_Time_min": 55.0
        }
    ])

    for model_name, cls in [("tabpfn", TabPFNPredictor), ("xgboost", XGBoostPredictor), ("linear_regression", LinearRegressionPredictor)]:
        pred_instance = cls()
        pred_instance.fit(df_sample, df_sample["Delivery_Time_min"])
        assert pred_instance.is_fitted is True

        preds = pred_instance.predict(df_sample)
        assert len(preds) == 2
        assert preds[0] > 0
        assert preds[1] > preds[0]  # Longer distance & rain should take longer

        # Test late risk probability
        probs = pred_instance.predict_late_probability(df_sample, np.array([40.0, 40.0]))
        assert len(probs) == 2
        assert 0.0 <= probs[0] <= 1.0

def test_agent_json_tool_extraction():
    """Test JSON tool call extraction with markdown code fences and whitespace."""
    fenced_llm_output = """
    Sure, let me check the risky orders for you:
    ```json
    {
      "tool": "list_risky_orders",
      "args": {
        "threshold": 0.45
      }
    }
    ```
    I will scan the delivery queue.
    """
    extracted = extract_json(fenced_llm_output)
    assert extracted is not None
    assert extracted["tool"] == "list_risky_orders"
    assert extracted["args"]["threshold"] == 0.45

def test_agent_heuristic_routing_hindi_query():
    """Test query 'Aaj kaunsi orders late hongi?' properly maps to list_risky_orders tool."""
    user_query = "Aaj kaunsi orders late hongi?"
    route = heuristic_fallback_router(user_query)
    assert route is not None
    assert route["tool"] == "list_risky_orders"
    assert "threshold" in route["args"]

def test_tools_draft_message():
    """Test WhatsApp delay notification generation in Hinglish, Hindi, and English."""
    sample_order = {
        "order_id": "DB-10042",
        "customer_name": "Aarav Sharma",
        "expected_delay_min": 18,
        "traffic_level": "heavy evening traffic"
    }

    # Hinglish
    draft_hinglish = tool_draft_message(sample_order, language="Hinglish")
    assert "Aarav Sharma" in draft_hinglish["draft_text"]
    assert "DB-10042" in draft_hinglish["draft_text"]
    assert "delay" in draft_hinglish["draft_text"].lower()

    # Hindi
    draft_hindi = tool_draft_message(sample_order, language="Hindi")
    assert "नमस्ते" in draft_hindi["draft_text"]
    assert "आर्डर" in draft_hindi["draft_text"]

    # English
    draft_english = tool_draft_message(sample_order, language="English")
    assert "Hello Aarav Sharma" in draft_english["draft_text"]
    assert "delay" in draft_english["draft_text"]

@pytest.mark.asyncio
async def test_workflow_happy_path():
    """Test full Temporal activities execution pipeline (load -> predict -> flag -> explain -> draft)."""
    orders = await load_orders_activity(limit=10)
    assert len(orders) > 0

    predictions = await predict_all_activity(orders)
    assert len(predictions) == len(orders)
    assert "eta_minutes" in predictions[0]
    assert "late_risk_probability" in predictions[0]

    risky = await flag_risky_activity({"predictions": predictions, "threshold": 0.30})
    assert isinstance(risky, list)

    explanation = await explain_activity(risky)
    assert "summary" in explanation

    drafts = await draft_messages_activity(risky)
    assert len(drafts) == len(risky)
    if drafts:
        assert "message_text" in drafts[0]

def test_api_endpoints():
    """Test FastAPI /predict, /chat, and /run-dispatch endpoints."""
    from starlette.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 1. Test /predict
    pred_payload = {
        "Distance_km": 4.5,
        "Weather": "Clear",
        "Traffic_Level": "Medium",
        "Time_of_Day": "Evening",
        "Vehicle_Type": "Scooter",
        "Preparation_Time_min": 15.0,
        "Courier_Experience_yrs": 2.5,
        "promised_time_min": 40.0
    }
    r = client.post("/predict", json=pred_payload)
    assert r.status_code == 200
    res = r.json()
    assert "eta_minutes" in res
    assert "late_risk_probability" in res
    assert res["eta_minutes"] > 0

    # 2. Test /chat with "Aaj kaunsi orders late hongi?"
    chat_payload = {"message": "Aaj kaunsi orders late hongi?"}
    r_chat = client.post("/chat", json=chat_payload)
    assert r_chat.status_code == 200
    chat_res = r_chat.json()
    assert "reply" in chat_res
    assert chat_res["tool_called"] == "list_risky_orders"

    # 3. Test /run-dispatch
    r_dispatch = client.post("/run-dispatch", json={"late_threshold_probability": 0.45})
    assert r_dispatch.status_code == 200
    disp_res = r_dispatch.json()
    assert "workflow_id" in disp_res
    assert disp_res["status"] == "RUNNING"

