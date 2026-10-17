"""
app/main.py
FastAPI application for DabbaETA:
- Local-first delivery-time prediction + dispatch assistant
- Sentry tracing with custom spans
- SQLite / memory storage
- TabPFN, XGBoost, LinearRegression predictor integration
- Gemma agent tool routing
- Temporal dispatch workflow integration
"""

import os
import io
import sys
import uuid
import json
import time
import sqlite3
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import numpy as np
import sentry_sdk

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.schemas import (
    OrderSchema,
    OrderBatchPredictRequest,
    PredictionResult,
    UploadResponse,
    BenchmarkResponse,
    ChatRequest,
    ChatResponse,
    DispatchStartRequest,
    DispatchStatusResponse,
)
from app.predictor import get_predictor, TabPFNPredictor
from app.baselines import XGBoostPredictor, LinearRegressionPredictor
from app.agent import process_agent_chat
from app.tools import get_orders_df, tool_draft_message, DB_PATH, CSV_PATH
from scripts.benchmark import BENCHMARK_JSON, BENCHMARK_PNG, run_benchmarks

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dabbaeta.api")

# Initialize Sentry
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=1.0,
        release="dabbaeta@1.0.0",
        environment=os.getenv("ENVIRONMENT", "development")
    )
    logger.info("Sentry initialized for FastAPI.")

app = FastAPI(
    title="DabbaETA API",
    description="Local-first delivery-time prediction + dispatch assistant for tiffins & bakeries",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workflow Job Storage (for local fallback and tracking)
WORKFLOW_JOBS: Dict[str, Dict[str, Any]] = {}

def init_db():
    """Initializes SQLite database if not present."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            Order_ID TEXT PRIMARY KEY,
            Customer_Name TEXT,
            Customer_Phone TEXT,
            Delivery_Location TEXT,
            Distance_km REAL,
            Weather TEXT,
            Traffic_Level TEXT,
            Time_of_Day TEXT,
            Vehicle_Type TEXT,
            Preparation_Time_min REAL,
            Courier_Experience_yrs REAL,
            promised_time_min REAL,
            Delivery_Time_min REAL,
            is_late INTEGER
        )
    """)
    conn.commit()

    # If DB is empty but CSV exists, seed from CSV
    cursor.execute("SELECT COUNT(*) FROM orders")
    count = cursor.fetchone()[0]
    if count == 0 and os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            # Ensure required columns
            for col in ["Customer_Name", "Customer_Phone", "Delivery_Location", "promised_time_min", "is_late"]:
                if col not in df.columns:
                    if col == "promised_time_min":
                        df[col] = 40.0
                    elif col == "is_late":
                        df[col] = (df.get("Delivery_Time_min", 0) > 40).astype(int)
                    else:
                        df[col] = "N/A"
            df.to_sql("orders", conn, if_exists="append", index=False)
            logger.info(f"Seeded SQLite with {len(df)} orders from CSV.")
        except Exception as e:
            logger.warning(f"Could not seed SQLite from CSV: {e}")
    conn.close()

init_db()

# Warm up default predictor
try:
    if os.path.exists(CSV_PATH):
        df_train = pd.read_csv(CSV_PATH).head(300)
        p = get_predictor("tabpfn")
        if not p.is_fitted:
            p.fit(df_train, df_train["Delivery_Time_min"])
            logger.info("TabPFN predictor warmed up.")
except Exception as e:
    logger.warning(f"Predictor warm-up skipped: {e}")


@app.get("/")
def read_root():
    index_file = os.path.join(ROOT_DIR, "frontend", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "app": "DabbaETA",
        "description": "Local-first delivery ETA & dispatch assistant",
        "status": "online",
        "docs": "/docs",
        "endpoints": ["/orders", "/upload", "/predict", "/benchmark", "/chat", "/run-dispatch"]
    }

@app.get("/api")
def read_api_info():
    return {
        "app": "DabbaETA",
        "description": "Local-first delivery ETA & dispatch assistant",
        "status": "online",
        "docs": "/docs",
        "endpoints": ["/orders", "/upload", "/predict", "/benchmark", "/chat", "/run-dispatch"]
    }


@app.get("/orders")
def get_orders(limit: int = 50, model: str = "tabpfn"):
    """Fetches pending orders annotated with current ETA and late risk probability."""
    df = get_orders_df(limit=limit)
    if df.empty:
        return []

    predictor = get_predictor(model)
    etas = predictor.predict(df)
    promised_times = df["promised_time_min"].fillna(40.0).values
    probs = predictor.predict_late_probability(df, promised_times)

    results = []
    for idx, row in df.iterrows():
        eta = float(etas[idx])
        promised = float(promised_times[idx])
        prob = float(probs[idx])
        is_late = bool(eta > promised or prob >= 0.5)

        results.append({
            "Order_ID": row.get("Order_ID", f"DB-{10000+idx}"),
            "Customer_Name": row.get("Customer_Name", "Customer"),
            "Customer_Phone": row.get("Customer_Phone", "+91 98000 00000"),
            "Delivery_Location": row.get("Delivery_Location", "Local Destination"),
            "Distance_km": float(row.get("Distance_km", 4.0)),
            "Weather": str(row.get("Weather", "Clear")),
            "Traffic_Level": str(row.get("Traffic_Level", "Medium")),
            "Time_of_Day": str(row.get("Time_of_Day", "Evening")),
            "Vehicle_Type": str(row.get("Vehicle_Type", "Scooter")),
            "Preparation_Time_min": float(row.get("Preparation_Time_min", 15.0)),
            "Courier_Experience_yrs": float(row.get("Courier_Experience_yrs", 2.0)),
            "promised_time_min": round(promised, 1),
            "eta_minutes": round(eta, 1),
            "delay_minutes": max(0.0, round(eta - promised, 1)),
            "late_risk_probability": round(prob, 2),
            "is_late_predicted": is_late,
            "risk_badge": "HIGH" if prob >= 0.65 else ("MODERATE" if prob >= 0.35 else "LOW")
        })
    return results


@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Accepts CSV, validates schema against required columns,
    adds missing promised_time_min/is_late, and stores into SQLite and data/orders.csv.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # Validate essential schema
    required_cols = ["Distance_km", "Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns in CSV: {missing}")

    if "Order_ID" not in df.columns:
        df["Order_ID"] = [f"DB-{10000 + i}" for i in range(len(df))]
    if "promised_time_min" not in df.columns:
        df["promised_time_min"] = 40.0
    if "Delivery_Time_min" not in df.columns:
        df["Delivery_Time_min"] = None
    if "is_late" not in df.columns:
        if df["Delivery_Time_min"].notnull().all():
            df["is_late"] = (df["Delivery_Time_min"] > df["promised_time_min"]).astype(int)
        else:
            df["is_late"] = 0

    for optional_col in ["Customer_Name", "Customer_Phone", "Delivery_Location"]:
        if optional_col not in df.columns:
            df[optional_col] = "Local Patron"

    # Store in SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

    # Also persist to data/orders.csv
    df.to_csv(CSV_PATH, index=False)

    # Retrain/fit predictor with new data
    try:
        p = get_predictor("tabpfn")
        if df["Delivery_Time_min"].notnull().sum() > 20:
            df_valid = df.dropna(subset=["Delivery_Time_min"])
            p.fit(df_valid, df_valid["Delivery_Time_min"])
    except Exception as err:
        logger.warning(f"Predictor re-fit failed on upload: {err}")

    preview = df.head(5).to_dict(orient="records")
    return UploadResponse(
        status="success",
        message=f"Successfully loaded and validated {len(df)} orders.",
        rows_processed=len(df),
        columns=list(df.columns),
        sample_preview=preview
    )


@app.post("/predict", response_model=PredictionResult)
def predict_eta(order: OrderSchema, model: str = Query("tabpfn", description="tabpfn, xgboost, or linear_regression")):
    """
    Returns ETA (minutes) + late-risk probability for given order features.
    """
    predictor = get_predictor(model)
    df = pd.DataFrame([order.model_dump()])

    t0 = time.time()
    eta = float(predictor.predict(df)[0])
    latency_ms = (time.time() - t0) * 1000.0

    promised = float(order.promised_time_min or 40.0)
    probs = predictor.predict_late_probability(df, np.array([promised]))
    late_prob = float(probs[0])
    is_late = bool(eta > promised or late_prob >= 0.5)

    res_std = getattr(predictor, "residual_std_", 3.5)
    ci = {
        "lower": round(max(5.0, eta - 1.96 * res_std), 1),
        "upper": round(eta + 1.96 * res_std, 1)
    }

    return PredictionResult(
        order_id=order.Order_ID or f"ORD-{uuid.uuid4().hex[:6].upper()}",
        eta_minutes=round(eta, 1),
        promised_time_min=round(promised, 1),
        late_risk_probability=round(late_prob, 3),
        is_late_predicted=is_late,
        delay_minutes=max(0.0, round(eta - promised, 1)),
        model_used=predictor.name,
        latency_ms=round(latency_ms, 2),
        confidence_interval=ci
    )


@app.get("/benchmark", response_model=BenchmarkResponse)
def get_benchmark(refresh: bool = False):
    """
    Trains on 80% / tests on 20%, returns MAE, RMSE, R2, and fit+predict latency
    for TabPFN vs XGBoost vs LinearRegression. Also includes small-data experiment (50, 100, 200).
    """
    if not refresh and os.path.exists(BENCHMARK_JSON):
        try:
            with open(BENCHMARK_JSON, "r") as f:
                data = json.load(f)
            data["chart_path"] = "/docs/benchmark.png"
            return data
        except Exception:
            pass

    # Run benchmark
    results = run_benchmarks()
    results["chart_path"] = "/docs/benchmark.png"
    return results


@app.get("/docs/benchmark.png")
def get_benchmark_chart():
    """Returns the generated benchmark chart image."""
    if not os.path.exists(BENCHMARK_PNG):
        run_benchmarks()
    return FileResponse(BENCHMARK_PNG, media_type="image/png")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Gemma-driven dispatch assistant chat with JSON tool routing.
    Queries like 'Aaj kaunsi orders late hongi?' are routed to list_risky_orders.
    """
    result = await process_agent_chat(request.message)
    return ChatResponse(
        reply=result["reply"],
        tool_called=result.get("tool_called"),
        tool_args=result.get("tool_args"),
        tool_result=result.get("tool_result")
    )


async def execute_local_dispatch_workflow(job_id: str, threshold: float, limit: int = 25):
    """Fallback in-process runner matching Temporal DispatchWorkflow semantics."""
    WORKFLOW_JOBS[job_id]["status"] = "RUNNING"
    try:
        from workflows.activities import (
            load_orders_activity,
            predict_all_activity,
            flag_risky_activity,
            explain_activity,
            draft_messages_activity
        )

        orders = await load_orders_activity(limit)
        predictions = await predict_all_activity(orders)
        risky_orders = await flag_risky_activity({"predictions": predictions, "threshold": threshold})
        explanation = await explain_activity(risky_orders)
        drafts = await draft_messages_activity(risky_orders)

        WORKFLOW_JOBS[job_id].update({
            "status": "COMPLETED",
            "total_orders": len(orders),
            "risky_orders_count": len(risky_orders),
            "results": predictions,
            "draft_messages": drafts,
            "explanation": explanation,
        })
    except Exception as err:
        logger.error(f"Dispatch workflow error in job {job_id}: {err}")
        WORKFLOW_JOBS[job_id]["status"] = "FAILED"
        WORKFLOW_JOBS[job_id]["error"] = str(err)


@app.post("/run-dispatch")
async def run_dispatch(req: Optional[DispatchStartRequest] = None, background_tasks: BackgroundTasks = None):
    """
    Triggers DispatchWorkflow (activities: load_orders -> predict_all -> flag_risky -> explain -> draft_messages).
    Uses Temporal if server is running; otherwise smoothly falls back to asynchronous in-process orchestrator.
    """
    threshold = req.late_threshold_probability if req else 0.45
    job_id = f"dispatch-{uuid.uuid4().hex[:8]}"

    WORKFLOW_JOBS[job_id] = {
        "workflow_id": job_id,
        "status": "STARTING",
        "total_orders": 0,
        "risky_orders_count": 0,
        "results": [],
        "draft_messages": [],
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # Attempt connecting to Temporal server
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    temporal_started = False
    try:
        from temporalio.client import Client
        from workflows.workflows import DispatchWorkflow, TASK_QUEUE_NAME

        # Fast connection attempt
        client = await asyncio.wait_for(Client.connect(temporal_host), timeout=1.5)
        handle = await client.start_workflow(
            DispatchWorkflow.run,
            {"threshold": threshold, "limit": 25},
            id=job_id,
            task_queue=TASK_QUEUE_NAME
        )
        WORKFLOW_JOBS[job_id]["status"] = "RUNNING_ON_TEMPORAL"
        WORKFLOW_JOBS[job_id]["temporal"] = True
        temporal_started = True
        logger.info(f"Started Temporal workflow {job_id}")
    except Exception as e:
        logger.info(f"Temporal server not detected at {temporal_host} ({e}). Running in-process workflow orchestrator.")

    if not temporal_started:
        if background_tasks:
            background_tasks.add_task(execute_local_dispatch_workflow, job_id, threshold)
        else:
            asyncio.create_task(execute_local_dispatch_workflow, job_id, threshold)

    return {
        "workflow_id": job_id,
        "status": "RUNNING",
        "message": "DispatchWorkflow successfully triggered.",
        "check_status_url": f"/run-dispatch/{job_id}"
    }


@app.get("/run-dispatch/{job_id}", response_model=DispatchStatusResponse)
async def get_dispatch_status(job_id: str):
    """Checks the status and fetches results/draft messages of a dispatch run."""
    job = WORKFLOW_JOBS.get(job_id)

    # Check Temporal server if job was started on Temporal
    if job and job.get("temporal"):
        try:
            from temporalio.client import Client
            temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
            client = await Client.connect(temporal_host)
            handle = client.get_workflow_handle(job_id)
            desc = await handle.describe()
            if desc.status.name == "COMPLETED":
                result = await handle.result()
                job.update({
                    "status": "COMPLETED",
                    "total_orders": result.get("total_orders", 0),
                    "risky_orders_count": result.get("risky_orders_count", 0),
                    "results": result.get("predictions", []),
                    "draft_messages": result.get("draft_messages", []),
                })
        except Exception:
            pass

    if not job:
        raise HTTPException(status_code=404, detail=f"Workflow job {job_id} not found.")

    return DispatchStatusResponse(
        workflow_id=job["workflow_id"],
        status=job["status"],
        total_orders=job.get("total_orders", 0),
        risky_orders_count=job.get("risky_orders_count", 0),
        results=job.get("results"),
        draft_messages=job.get("draft_messages"),
        error=job.get("error")
    )

# Static frontend mount
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
    @app.get("/app")
    def serve_frontend_page():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

