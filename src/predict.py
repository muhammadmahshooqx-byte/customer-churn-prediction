from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .preprocessing import prepare_features
except ImportError:
    from preprocessing import prepare_features


def predict(customer: dict, model_path: str = "models/churn_model.joblib") -> dict:
    artifact = joblib.load(model_path)
    features = prepare_features(pd.DataFrame([customer]))
    probability = float(artifact["pipeline"].predict_proba(features)[:, 1][0])
    return {"churn_probability": probability, "risk": "High" if probability >= .5 else "Low"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict churn risk for one customer CSV row.")
    parser.add_argument("csv", help="CSV containing one customer row")
    parser.add_argument("--model", default="models/churn_model.joblib")
    args = parser.parse_args()
    row = pd.read_csv(Path(args.csv)).iloc[0].to_dict()
    print(predict(row, args.model))
