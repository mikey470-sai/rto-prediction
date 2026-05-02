"""
RTO Prediction Engine — API Unit Tests
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.main import app
client = TestClient(app)

VALID_PAYLOAD = {
    "payment_mode":         "COD",
    "seller_tier":          "SILVER",
    "customer_city_tier":   2,
    "distance_km":          450,
    "product_category":     "FASHION",
    "order_value":          799.0,
    "customer_rto_history": 1,
    "delivery_attempts":    1,
    "day_of_week":          3,
    "is_weekend":           0,
    "pincode_rto_rate":     0.18,
    "seller_rto_rate":      0.12,
}


# ── Mock model fixture ────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_model():
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.35, 0.65]])
    with patch("api.main._model", mock), \
         patch("api.main._feature_cols", list(VALID_PAYLOAD.keys())):
        yield mock


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_happy_path():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["rto_probability"] <= 1.0
    assert body["rto_flag"] is True          # prob=0.65 → flag True
    assert body["risk_band"] == "HIGH"
    assert "latency_ms" in body


def test_predict_low_risk_prepaid():
    from unittest.mock import patch
    import numpy as np
    low_risk_mock = MagicMock()
    low_risk_mock.predict_proba.return_value = np.array([[0.90, 0.10]])
    with patch("api.main._model", low_risk_mock):
        payload = {**VALID_PAYLOAD, "payment_mode": "PREPAID"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["risk_band"] == "LOW"
        assert r.json()["rto_flag"] is False


def test_predict_invalid_payment_mode():
    payload = {**VALID_PAYLOAD, "payment_mode": "CASH"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_order_value_zero():
    payload = {**VALID_PAYLOAD, "order_value": 0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_city_tier():
    payload = {**VALID_PAYLOAD, "customer_city_tier": 5}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "model_loaded" in r.json()