# AI-Powered Business Intelligence & Decision Support System

An end-to-end business intelligence system that analyses e-commerce data,
segments customers, predicts revenue and automatically generates plain
English business insights and recommendations.

Built on the Olist e-commerce dataset simulating a Nigerian e-commerce
environment similar to Jumia.

---

## Project Structure

| File | Description |
|------|-------------|
| `BI_DECISION_SUPPORT.ipynb` | Main notebook — all phases |
| `pipeline.py` | Automation pipeline — drop a CSV to trigger |
| `BI_DECISION_SUPPORT_VISUAL.pbix` | Power BI dashboard |
| `business_query.sql` | SQL business queries |
| `churn_model.joblib` | Trained churn prediction model |
| `sales_model.joblib` | Trained sales forecasting model |
| `demand_model.joblib` | Trained demand prediction model |

---

## What This System Does

- Loads and cleans 9 raw e-commerce datasets
- Performs exploratory data analysis and visualization
- Segments customers using RFM analysis (Recency, Frequency, Monetary)
- Predicts customer churn with 89% recall
- Forecasts monthly revenue with 99.95% R2 score
- Predicts product demand with Random Forest Regression
- Generates plain English business insights automatically using Groq AI
- Assigns tailored recommendations to every customer based on their segment
- Stores all data in MySQL database
- Automates the entire pipeline — drop a CSV file to trigger a full run
- Visualizes everything in a Power BI dashboard

---

## Phases Completed

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Setup | ✅ |
| 2 | Data Loading & Overview | ✅ |
| 3 | Data Cleaning | ✅ |
| 4 | EDA & Visualization | ✅ |
| 5 | Customer Intelligence (RFM) | ✅ |
| 6 | Preprocessing | ✅ |
| 7 | Machine Learning | ✅ |
| 8 | AI Insight Generator | ✅ |
| 9 | Recommendation Engine | ✅ |
| 10 | SQL Integration | ✅ |
| 11 | Power BI Dashboard | ✅ |
| 12 | Automation Pipeline | ✅ |
| 13 | GitHub & Documentation | ✅ |

---

## ML Model Performance

| Model | Algorithm | Metric | Score |
|-------|-----------|--------|-------|
| Churn Prediction | Random Forest Classifier | Recall | 89% |
| Churn Prediction | Random Forest Classifier | ROC-AUC | 87% |
| Sales Forecasting | Random Forest Regressor | R2 Score | 99.95% |
| Demand Prediction | Random Forest Regressor | R2 Score | 53.51% |

---

## Customer Segments

| Segment | Count | Description |
|---------|-------|-------------|
| Regular | 33,920 (39.3%) | Occasional buyers |
| Needs Attention | 30,350 (35.16%) | Engagement dropping |
| Premium | 14,290 (16.55%) | High value, frequent buyers |
| Churned | 7,760 (8.99%) | Inactive customers |

---

## Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- MySQL + SQLAlchemy
- Groq AI API (llama-3.3-70b-versatile)
- Power BI
- Jupyter Notebook

---

## Setup Instructions

1. Clone the repository
2. Download the raw dataset from Kaggle:
   https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
3. Place all 9 CSV files in the project folder
4. Install dependencies:
