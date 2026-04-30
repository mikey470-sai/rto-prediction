"""
RTO Prediction Engine — FastAPI Service
Endpoints: POST /predict  |  GET /health  |  GET /metrics
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import os
import time
from typing import Optional

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RTO Prediction Engine",
    description="Predicts Return-to-Origin probability for e-commerce shipments.",
    version="1.0.0",
)

MODEL_PATH   = os.getenv("MODEL_PATH",   "models/artifacts/lgbm_model.pkl")
FEATURE_PATH = os.getenv("FEATURE_PATH", "models/artifacts/feature_cols.pkl")

# Lazy-load model at startup
_model        = None
_feature_cols = None

def load_model():
    global _model, _feature_cols
    if _model is None:
        _model        = joblib.load(MODEL_PATH)
        _feature_cols = joblib.load(FEATURE_PATH)


# ── Schemas ───────────────────────────────────────────────────────────────────
PAYMENT_MODES   = {"COD", "PREPAID"}
SELLER_TIERS    = {"GOLD", "SILVER", "BRONZE"}
CATEGORIES      = {"ELECTRONICS", "FASHION", "HOME", "BEAUTY"}

class ShipmentRequest(BaseModel):
    payment_mode:         str   = Field(..., example="COD")
    seller_tier:          str   = Field(..., example="SILVER")
    customer_city_tier:   int   = Field(..., ge=1, le=3, example=2)
    distance_km:          int   = Field(..., ge=1, le=5000, example=450)
    product_category:     str   = Field(..., example="FASHION")
    order_value:          float = Field(..., gt=0, example=799.0)
    customer_rto_history: int   = Field(0, ge=0, le=20, example=1)
    delivery_attempts:    int   = Field(1, ge=1, le=5, example=1)
    day_of_week:          int   = Field(..., ge=0, le=6, example=3)
    is_weekend:           int   = Field(..., ge=0, le=1, example=0)
    pincode_rto_rate:     float = Field(..., ge=0, le=1, example=0.18)
    seller_rto_rate:      float = Field(..., ge=0, le=1, example=0.12)

    @validator("payment_mode")
    def validate_payment(cls, v):
        if v.upper() not in PAYMENT_MODES:
            raise ValueError(f"payment_mode must be one of {PAYMENT_MODES}")
        return v.upper()

    @validator("seller_tier")
    def validate_tier(cls, v):
        if v.upper() not in SELLER_TIERS:
            raise ValueError(f"seller_tier must be one of {SELLER_TIERS}")
        return v.upper()

    @validator("product_category")
    def validate_category(cls, v):
        if v.upper() not in CATEGORIES:
            raise ValueError(f"product_category must be one of {CATEGORIES}")
        return v.upper()


class PredictionResponse(BaseModel):
    rto_probability:  float
    rto_flag:         bool
    risk_band:        str   # LOW / MEDIUM / HIGH
    confidence:       str
    latency_ms:       float


# ── Helpers ───────────────────────────────────────────────────────────────────
_CATEGORY_MAP     = {c: i for i, c in enumerate(["BEAUTY", "ELECTRONICS", "FASHION", "HOME"])}
_PAYMENT_MAP      = {"COD": 0, "PREPAID": 1}
_TIER_MAP         = {"BRONZE": 0, "GOLD": 1, "SILVER": 2}

def encode_request(req: ShipmentRequest) -> list:
    return [
        _PAYMENT_MAP[req.payment_mode],
        _TIER_MAP[req.seller_tier],
        req.customer_city_tier,
        req.distance_km,
        _CATEGORY_MAP[req.product_category],
        req.order_value,
        req.customer_rto_history,
        req.delivery_attempts,
        req.day_of_week,
        req.is_weekend,
        req.pincode_rto_rate,
        req.seller_rto_rate,
    ]

def risk_band(prob: float) -> str:
    if prob < 0.25:  return "LOW"
    if prob < 0.50:  return "MEDIUM"
    return "HIGH"

def confidence(prob: float) -> str:
    dist = abs(prob - 0.5)
    if dist > 0.30: return "HIGH"
    if dist > 0.15: return "MEDIUM"
    return "LOW"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    if os.path.exists(MODEL_PATH):
        load_model()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "version": app.version,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: ShipmentRequest):
    if _model is None:
        try:
            load_model()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run models/train.py first.",
            )

    t0 = time.perf_counter()
    features = np.array([encode_request(req)])
    prob = float(_model.predict_proba(features)[0, 1])
    latency = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        rto_probability = round(prob, 4),
        rto_flag        = prob >= 0.5,
        risk_band       = risk_band(prob),
        confidence      = confidence(prob),
        latency_ms      = round(latency, 2),
    )

@app.get("/metrics")
def metrics():
    """Prometheus-style plaintext metrics (extend with prometheus-client)."""
    return {"uptime": "available", "model_loaded": _model is not None}
