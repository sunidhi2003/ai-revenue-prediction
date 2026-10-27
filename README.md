# AI-Powered Revenue Prediction System

A small, review-friendly machine-learning project that predicts revenue from a CSV dataset using a Flask API.

## Features
- CSV data ingestion
- Exploratory data analysis (EDA) and visualizations
- Compares a baseline model with Linear Regression and Random Forest
- Flask prediction API for one country or all countries combined
- Unit tests for API, model, and logging
- Basic performance logging
- Docker support

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/train_model.py
python app.py
```

Open `http://127.0.0.1:5000/health`.

## Predict
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"target_date":"2026-11-01","country":"United Kingdom"}'
```

Use `"country":"ALL"` to predict the combined revenue across countries.

## Run tests
```bash
pytest -q
```

## EDA
```bash
python scripts/eda.py
```
Charts are saved under `artifacts/`.

## Docker
```bash
docker build -t ai-revenue-prediction .
docker run -p 5000:5000 ai-revenue-prediction
```

## Dataset note
The included data generator creates **synthetic demo data**, not real business data. Replace `data/revenue.csv` with an appropriate licensed dataset for real-world use. Predictions are for demonstration and are not business advice.
