"""
RTO Prediction Engine — Model Training
Uses LightGBM with MLflow experiment tracking
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_experiment("rto-prediction")

# ── Feature engineering ───────────────────────────────────────────────────────
def generate_synthetic_data(n=50_000, seed=42):
    """Generate synthetic shipment data for demo purposes."""
    rng = np.random.default_rng(seed)

    data = pd.DataFrame({
        "payment_mode":         rng.choice(["COD", "PREPAID"], n, p=[0.6, 0.4]),
        "seller_tier":          rng.choice(["GOLD", "SILVER", "BRONZE"], n, p=[0.2, 0.5, 0.3]),
        "customer_city_tier":   rng.choice([1, 2, 3], n, p=[0.3, 0.4, 0.3]),
        "distance_km":          rng.integers(10, 2000, n),
        "product_category":     rng.choice(["ELECTRONICS", "FASHION", "HOME", "BEAUTY"], n),
        "order_value":          rng.lognormal(mean=6.5, sigma=1.2, size=n).round(2),
        "customer_rto_history": rng.integers(0, 5, n),
        "delivery_attempts":    rng.integers(1, 4, n),
        "day_of_week":          rng.integers(0, 7, n),
        "is_weekend":           rng.integers(0, 2, n),
        "pincode_rto_rate":     rng.uniform(0.05, 0.45, n).round(3),
        "seller_rto_rate":      rng.uniform(0.05, 0.35, n).round(3),
    })

    # Simulate realistic RTO probability
    rto_prob = (
        0.35 * (data["payment_mode"] == "COD").astype(float)
        + 0.15 * (data["customer_rto_history"] / 5)
        + 0.20 * data["pincode_rto_rate"]
        + 0.15 * data["seller_rto_rate"]
        + 0.10 * (data["delivery_attempts"] / 3)
        - 0.05 * (data["seller_tier"] == "GOLD").astype(float)
        + rng.normal(0, 0.05, n)
    ).clip(0, 1)

    data["is_rto"] = (rng.random(n) < rto_prob).astype(int)
    return data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals and return feature matrix."""
    cat_cols = ["payment_mode", "seller_tier", "product_category"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("📦 Loading data ...")
    df = generate_synthetic_data()
    df = build_features(df)

    feature_cols = [c for c in df.columns if c != "is_rto"]
    X, y = df[feature_cols], df["is_rto"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "n_estimators":     500,
        "learning_rate":    0.05,
        "num_leaves":       63,
        "min_child_samples": 50,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "class_weight":     "balanced",
        "random_state":     42,
        "n_jobs":           -1,
    }

    print("🚀 Training LightGBM ...")
    with mlflow.start_run():
        mlflow.log_params(params)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        preds  = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, probas)

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("test_size", len(X_test))
        mlflow.lightgbm.log_model(model, "model")

        print(f"\n✅ AUC: {auc:.4f}")
        print(classification_report(y_test, preds, target_names=["DELIVERED", "RTO"]))

        # Save locally for the API
        os.makedirs("models/artifacts", exist_ok=True)
        joblib.dump(model,        "models/artifacts/lgbm_model.pkl")
        joblib.dump(feature_cols, "models/artifacts/feature_cols.pkl")
        print("💾 Model saved to models/artifacts/")

    return model, feature_cols


if __name__ == "__main__":
    train()
