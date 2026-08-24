from pathlib import Path

from src.predict import predict
from src.train_model import train


def test_training_and_prediction(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics = train(output_path=str(model_path))
    assert model_path.exists()
    assert 0 <= metrics["roc_auc"] <= 1
    result = predict({"Contract": "Month-to-month", "tenure": 3, "MonthlyCharges": 95, "TotalCharges": 285, "InternetService": "Fiber optic"}, str(model_path))
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk"] in {"High", "Low"}
