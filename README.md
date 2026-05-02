# 🚚 RTO Prediction Engine

> Predicts **Return-to-Origin (RTO)** probability for e-commerce shipments before dispatch — reducing logistics costs and improving delivery success rates.

[![CI/CD](https://github.com/mikey470-sai/rto-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/mikey470-sai/rto-prediction/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-green)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![Docker](https://img.shields.io/badge/docker-ready-blue)

---

## 🌐 Live Links

| What | Link |
|------|------|
| ⚡ Live API (Swagger UI) | https://rto-prediction-production.up.railway.app/docs |
| 🎨 Interactive Demo | [rto-demo.html](./rto-demo.html) — open in browser |
| 📊 MLflow Dashboard | Run `mlflow ui` locally → http://localhost:5000 |
| 🐙 GitHub Repo | https://github.com/mikey470-sai/rto-prediction |

---

## 📌 Problem

RTO (Return to Origin) is one of the biggest cost centers in Indian e-commerce logistics. When a shipment comes back undelivered, it costs 2–3× the original shipping fee. Industry RTO rates range from **20–40%** for COD orders.

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
rto-demo.html (Frontend) ──► FastAPI /predict ──► LightGBM Model
                                    │                     │
                                    │                     ▼
                                    │              RTO Probability
                                    │              Risk Band (LOW/MED/HIGH)
                                    ▼
                              Response (<5ms p99)

MLflow Tracking Server
    └── Experiment runs, AUC metrics, model registry
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.707 |
| Precision (RTO) | ~71% |
| Recall (RTO) | ~68% |
| Latency p99 | <5ms |
| Training data | 50,000 synthetic shipments |

---

## 🎨 Interactive Demo

Open `rto-demo.html` in your browser for a beautiful UI to test predictions:

- ✅ 3 preset scenarios — High Risk, Medium, Safe
- 📊 Live probability meter with color coding
- 🔍 Feature breakdown — why each prediction was made
- ⚡ Calls the real Railway API — not a simulation
- 📊 Links to MLflow dashboard and GitHub

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/mikey470-sai/rto-prediction.git
cd rto-prediction
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python models/train.py
# → Saves model to models/artifacts/
# → Logs experiment to MLflow
```

### 3. Start the API
```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### 4. Open Demo
```bash
# Just double-click rto-demo.html in your browser!
```

### 5. View MLflow
```bash
mlflow ui
# → http://localhost:5000
```

### 6. Run Tests
```bash
pytest tests/ -v
# → 7/7 tests passed
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
  "rto_probability": 0.4982,
  "rto_flag": false,
  "risk_band": "MEDIUM",
  "confidence": "LOW",
  "latency_ms": 0.78
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
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
│   └── test_api.py           # 7 unit tests (all passing)
├── .github/
│   └── workflows/
│       └── ci.yml            # Train → Test → Docker build
├── rto-demo.html             # 🎨 Interactive demo frontend
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🐳 Docker

```bash
docker build -t rto-prediction .
docker run -p 8000:8000 rto-prediction
```

---

## 🔑 Key Features

- **LightGBM** — fast gradient boosting, handles class imbalance
- **MLflow** — full experiment tracking and model registry
- **FastAPI** — async REST API with Pydantic validation
- **GitHub Actions** — CI/CD: train → test → Docker build
- **Railway** — cloud deployment with public URL
- **Interactive Demo** — beautiful HTML frontend calling real API

---

## 📄 License

MIT
