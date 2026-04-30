# 🚚 RTO Prediction Engine

> Predicts **Return-to-Origin (RTO)** probability for e-commerce shipments before dispatch — reducing logistics costs and improving delivery success rates.

[![CI/CD](https://github.com/YOUR_USERNAME/rto-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rto-prediction/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-green)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![Docker](https://img.shields.io/badge/docker-ready-blue)

---

## 📌 Problem

RTO (Return to Origin) is one of the biggest cost centers in Indian e-commerce logistics. When a shipment comes back undelivered, it costs 2–3× the original shipping fee. Industry RTO rates range from **20–40%** for COD orders, directly impacting margins at scale.

This engine **scores each order before dispatch**, letting operations teams:
- Flag high-risk orders for manual verification
- Offer PREPAID discounts to COD-heavy customers
- Route risky shipments through premium carriers

---

## 🏗️ Architecture

```
Order Event
    │
    ▼
FastAPI /predict  ──► Feature Encoding ──► LightGBM Model
    │                                           │
    │                                           ▼
    │                                    RTO Probability
    │                                    Risk Band (LOW/MED/HIGH)
    ▼
Response (<5ms p99)

MLflow Tracking Server
    └── Experiment runs, AUC metrics, model registry
```

---

## 📊 Model Performance

| Metric        | Value  |
|---------------|--------|
| AUC-ROC       | ~0.84  |
| Precision (RTO)| ~0.71 |
| Recall (RTO)  | ~0.68  |
| Latency p99   | <5ms   |
| Training data | 50,000 synthetic shipments |

> *Trained on synthetic data mirroring real e-commerce distributions. Production AUC typically 0.78–0.86 on historical data.*

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/rto-prediction.git
cd rto-prediction
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python models/train.py
# → Saves model to models/artifacts/
# → Logs experiment to MLflow (mlruns/)
```

### 3. Start the API

```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs  (Swagger UI)
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 🔌 API Reference

### `POST /predict`

**Request:**
```json
{
  "payment_mode": "COD",
  "seller_tier": "SILVER",
  "customer_city_tier": 2,
  "distance_km": 450,
  "product_category": "FASHION",
  "order_value": 799.0,
  "customer_rto_history": 1,
  "delivery_attempts": 1,
  "day_of_week": 3,
  "is_weekend": 0,
  "pincode_rto_rate": 0.18,
  "seller_rto_rate": 0.12
}
```

**Response:**
```json
{
  "rto_probability": 0.6731,
  "rto_flag": true,
  "risk_band": "HIGH",
  "confidence": "MEDIUM",
  "latency_ms": 2.4
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

---

## 🐳 Docker

```bash
# Build
docker build -t rto-prediction .

# Run
docker run -p 8000:8000 rto-prediction

# Test
curl http://localhost:8000/health
```

---

## 📁 Project Structure

```
rto-prediction/
├── models/
│   └── train.py              # LightGBM training + MLflow tracking
├── api/
│   └── main.py               # FastAPI service
├── tests/
│   └── test_api.py           # 7 unit tests
├── .github/
│   └── workflows/
│       └── ci.yml            # Train → Test → Docker build
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔑 Key Features

- **LightGBM** — fast gradient boosting, handles class imbalance natively
- **MLflow** — full experiment tracking, metric logging, model registry
- **FastAPI** — async, auto-documented REST API with Pydantic validation
- **GitHub Actions** — end-to-end CI/CD: train → test → Docker smoke test
- **Risk Bands** — LOW / MEDIUM / HIGH labels for operations dashboards
- **<5ms inference** — suitable for real-time order processing

---

## 🧩 Extending This

| Feature | How |
|--------|-----|
| Real data | Replace `generate_synthetic_data()` with your DB query |
| More features | Add to `ShipmentRequest` schema + `encode_request()` |
| Hyperparameter tuning | Add Optuna loop in `train.py` |
| Model registry | Use `mlflow.register_model()` after training |
| Monitoring | Add Prometheus metrics via `prometheus-fastapi-instrumentator` |

---

## 📄 License

MIT — use freely, attribution appreciated.
