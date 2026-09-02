# Customer Churn Risk Prediction


A reproducible telecom churn baseline that cleans the IBM/Kaggle Telco Customer Churn CSV, engineers service and charge features, trains a class-weighted Random Forest, and exposes predictions through a CLI or Flask form.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train_model.py
pytest
python app.py
```

The training command uses deterministic demo data when no CSV is present, so the project runs immediately. For the real dataset, place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/` or pass `--input path/to/file.csv`. The expected target column is `Churn` with `Yes`/`No` values.

## Project structure


- `data/raw/`: source CSV files
- `data/processed/`: optional exported datasets
- `notebooks/`: EDA notebooks
- `src/preprocessing.py`: loading, missing-value preparation, and feature engineering
- `src/train_model.py`: split, pipeline training, and metrics
- `src/predict.py`: single-customer inference
- `app.py`: optional local Flask interface
- `models/`: generated model artifacts

The generated artifact includes the full preprocessing pipeline and evaluation metrics. Accuracy and ROC-AUC are printed after training rather than hard-coded, because they depend on the supplied dataset.

## Future improvements

- Add SHAP values for per-customer explainability.
- Compare additional ensemble models and tune the recall/precision tradeoff.
- Deploy the Flask interface as a live web app.
