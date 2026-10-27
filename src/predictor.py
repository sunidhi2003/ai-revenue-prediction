import os
import joblib
import pandas as pd
from src.features import make_features

MODEL_PATH = os.path.join("models", "revenue_model.joblib")
COUNTRIES_PATH = os.path.join("models", "countries.joblib")

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COUNTRIES_PATH):
        raise FileNotFoundError("Model artifacts not found; train the model first.")
    return joblib.load(MODEL_PATH), joblib.load(COUNTRIES_PATH)

def predict_revenue(model, target_date, country, countries):
    if country.upper() == "ALL":
        predictions = []
        for item in countries:
            row = make_features(target_date, item, countries)
            value = float(model.predict(pd.DataFrame([row]))[0])
            predictions.append({"country": item, "predicted_revenue": round(max(0.0, value), 2)})
        total = round(sum(x["predicted_revenue"] for x in predictions), 2)
        return {"target_date": target_date, "country": "ALL", "predicted_revenue": total, "breakdown": predictions}
    row = make_features(target_date, country, countries)
    value = float(model.predict(pd.DataFrame([row]))[0])
    return {"target_date": target_date, "country": country, "predicted_revenue": round(max(0.0, value), 2)}
