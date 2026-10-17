"""
app/schemas.py
Pydantic schemas for DabbaETA API and Agent interactions.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class OrderSchema(BaseModel):
    Order_ID: Optional[str] = Field(default=None, description="Unique identifier for the order")
    Customer_Name: Optional[str] = Field(default="Valued Customer", description="Customer full name")
    Customer_Phone: Optional[str] = Field(default="+91 98000 00000", description="Customer contact number")
    Delivery_Location: Optional[str] = Field(default="Local Destination", description="Delivery drop address")
    Distance_km: float = Field(..., ge=0.1, le=50.0, description="Delivery distance in kilometers")
    Weather: str = Field(default="Clear", description="Weather condition (Clear, Rainy, Foggy, Windy, Stormy)")
    Traffic_Level: str = Field(default="Medium", description="Traffic condition (Low, Medium, High, Jam)")
    Time_of_Day: str = Field(default="Evening", description="Time of day (Morning, Afternoon, Evening, Night)")
    Vehicle_Type: str = Field(default="Scooter", description="Vehicle type (Bicycle, Scooter, Motorcycle, Electric_Bike)")
    Preparation_Time_min: float = Field(default=15.0, ge=0.0, le=120.0, description="Kitchen/bakery prep time in minutes")
    Courier_Experience_yrs: float = Field(default=2.0, ge=0.0, le=30.0, description="Rider experience in years")
    promised_time_min: Optional[float] = Field(default=40.0, ge=5.0, description="Committed delivery window in minutes")
    Delivery_Time_min: Optional[float] = Field(default=None, description="Actual delivery time if completed")
    is_late: Optional[int] = Field(default=None, description="1 if actual delivery exceeded promised time, else 0")

class OrderBatchPredictRequest(BaseModel):
    orders: List[OrderSchema]
    model_name: Optional[str] = Field(default="tabpfn", description="tabpfn, xgboost, or linear_regression")

class PredictionResult(BaseModel):
    order_id: str
    eta_minutes: float
    promised_time_min: float
    late_risk_probability: float
    is_late_predicted: bool
    delay_minutes: float
    model_used: str
    latency_ms: float
    confidence_interval: Optional[Dict[str, float]] = None

class UploadResponse(BaseModel):
    status: str
    message: str
    rows_processed: int
    columns: List[str]
    sample_preview: List[Dict[str, Any]]

class BenchmarkMetric(BaseModel):
    mae: float
    rmse: float
    r2: float
    fit_time_sec: float
    predict_time_sec: float
    sample_size: int

class BenchmarkResponse(BaseModel):
    full_dataset: Dict[str, BenchmarkMetric]
    small_data_experiments: Dict[str, Dict[str, BenchmarkMetric]]
    timestamp: str
    chart_path: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    reply: str
    tool_called: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None

class DispatchStartRequest(BaseModel):
    late_threshold_probability: Optional[float] = Field(default=0.45, ge=0.0, le=1.0)
    orders: Optional[List[OrderSchema]] = None

class DispatchStatusResponse(BaseModel):
    workflow_id: str
    status: str
    total_orders: int
    risky_orders_count: int
    results: Optional[List[Dict[str, Any]]] = None
    draft_messages: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
