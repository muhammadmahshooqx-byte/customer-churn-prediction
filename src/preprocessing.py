from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]
CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return make_demo_data()


def make_demo_data(rows: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[.55, .25, .20])
    tenure = rng.integers(0, 73, rows)
    monthly = np.round(rng.normal(70, 25, rows).clip(18, 125), 2)
    total = np.round((monthly * tenure * rng.normal(1, .08, rows)).clip(0), 2)
    logit = -1.0 + (contract == "Month-to-month") * 1.0 - tenure * .018 + (monthly > 85) * .5
    churn = np.where(rng.random(rows) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return pd.DataFrame({
        "customerID": [f"DEMO-{index:05d}" for index in range(rows)],
        "gender": rng.choice(["Male", "Female"], rows), "SeniorCitizen": rng.choice([0, 1], rows, p=[.84, .16]),
        "Partner": rng.choice(["Yes", "No"], rows), "Dependents": rng.choice(["Yes", "No"], rows, p=[.3, .7]),
        "PhoneService": "Yes", "MultipleLines": rng.choice(["Yes", "No", "No phone service"], rows),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], rows, p=[.35, .45, .2]),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], rows),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], rows),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], rows),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], rows),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], rows),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], rows),
        "Contract": contract, "PaperlessBilling": rng.choice(["Yes", "No"], rows),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], rows),
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total, TARGET: churn,
    })


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
    data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        if column not in data:
            data[column] = 0 if column == "SeniorCitizen" else "No"
    data["tenure_bucket"] = pd.cut(data["tenure"], bins=[-1, 6, 24, 48, np.inf], labels=["0-6", "7-24", "25-48", "49+"])
    service_columns = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for column in service_columns:
        if column not in data:
            data[column] = "No"
    data["total_services"] = data[service_columns].eq("Yes").sum(axis=1)
    data["monthly_to_total_ratio"] = data["MonthlyCharges"] / (data["TotalCharges"] + 1)
    return data.drop(columns=[column for column in DROP_COLUMNS if column in data], errors="ignore")


def split_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = prepare_features(frame)
    target = data.pop(TARGET).map({"Yes": 1, "No": 0, "Churned": 1, "Retained": 0})
    return data, target.astype(int)
