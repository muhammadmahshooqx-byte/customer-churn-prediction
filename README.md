# Customer Churn Prediction System

A machine learning system that predicts which telecom customers are likely to churn, enabling proactive retention strategies before customers leave.

## Problem

Customer acquisition costs significantly more than retention. Telecom providers lose revenue when customers churn without warning, and manual identification of at-risk customers is inconsistent and reactive. This project builds a predictive model that flags high-risk customers in advance, giving retention teams a data-driven head start.

## Dataset

- **Source:** Telco Customer Churn dataset (Kaggle / IBM sample dataset — update with actual source)
- **Size:** ~7,000 customer records (update once finalized)
- **Features:** Customer demographics, account tenure, contract type, monthly charges, total charges, service subscriptions (internet, phone, streaming, tech support), payment method
- **Target:** Binary churn label (Churned / Retained)
- **Preprocessing:** Handling missing values, encoding categorical features, feature scaling, class imbalance handling (SMOTE or class weighting)

## Approach

- Exploratory Data Analysis (EDA) to identify churn drivers — contract type, tenure, monthly charges, and service usage patterns
- Feature engineering: tenure buckets, total services subscribed, contract-to-charge ratio
- Model comparison across Logistic Regression, Random Forest, and Gradient Boosting (XGBoost/LightGBM)
- Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
- Evaluation using precision, recall, F1-score, and ROC-AUC — prioritizing recall on the churn class, since missing a true churner is costlier than a false alarm
- Feature importance analysis to surface the top drivers of churn for business interpretability

## Results

*(Fill in once trained — keep this format for consistency with other project write-ups)*

- Accuracy: TBD%
- ROC-AUC: TBD
- Top churn drivers identified: TBD

## Features

- Data pipeline: raw CSV → cleaned, encoded, model-ready dataset
- Trained classification model with saved artifact (`.pkl`)
- Feature importance visualization
- (Optional stretch goal) Simple web interface — upload customer data or enter details manually, receive churn risk prediction in real time

## Tech Stack

**ML/Data:** Python, Pandas, NumPy, scikit-learn, XGBoost
**Visualization:** Matplotlib, Seaborn
**Interface (optional):** Flask
**Notebook Environment:** Jupyter

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── models/
│   └── churn_model.pkl
├── app.py                 # optional web interface
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Clone the repo
git clone https://github.com/muhammadmahshooqx-byte/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run preprocessing + training
python src/train_model.py

# (Optional) Run the web interface
python app.py
```

## Future Improvements

- Deploy as a live web app with real-time prediction
- Add SHAP values for per-customer explainability
- Experiment with ensemble stacking for improved recall
- Build a retention-recommendation layer based on churn drivers

---

*Built as part of an ongoing portfolio to demonstrate end-to-end ML pipeline development — from raw data to a working, interpretable prediction system.*
