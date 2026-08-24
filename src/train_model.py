from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .preprocessing import NUMERIC_COLUMNS, load_data, split_target
except ImportError:
    from preprocessing import NUMERIC_COLUMNS, load_data, split_target


def build_pipeline(feature_names: list[str]) -> Pipeline:
    categorical = [name for name in feature_names if name not in NUMERIC_COLUMNS and name != "total_services" and name != "monthly_to_total_ratio"]
    numeric = [name for name in feature_names if name not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", min_samples_leaf=3, random_state=42, n_jobs=-1))])


def train(input_path: str | None = None, output_path: str = "models/churn_model.joblib") -> dict:
    features, target = split_target(load_data(input_path))
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.2, stratify=target, random_state=42)
    pipeline = build_pipeline(list(features.columns))
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, probabilities >= .5), "roc_auc": roc_auc_score(y_test, probabilities), "report": classification_report(y_test, probabilities >= .5, output_dict=True)}
    artifact = {"pipeline": pipeline, "feature_names": list(features.columns), "metrics": metrics}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the customer churn model.")
    parser.add_argument("--input", help="Path to the Telco CSV; uses deterministic demo data when omitted.")
    parser.add_argument("--output", default="models/churn_model.joblib")
    args = parser.parse_args()
    result = train(args.input, args.output)
    print(json.dumps({"accuracy": result["accuracy"], "roc_auc": result["roc_auc"]}, indent=2))
