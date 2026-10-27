from datetime import date
from flask import Flask, jsonify, request
from src.logging_config import get_logger
from src.predictor import load_artifacts, predict_revenue

app = Flask(__name__)
logger = get_logger(__name__)

try:
    MODEL, COUNTRIES = load_artifacts()
except FileNotFoundError:
    MODEL, COUNTRIES = None, []

@app.get("/")
def home():
    return jsonify({
        "name": "AI-Powered Revenue Prediction API",
        "endpoints": ["/health", "/predict"],
        "usage": "POST JSON to /predict with target_date (YYYY-MM-DD) and country (country name or ALL)"
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

@app.post("/predict")
def predict():
    if MODEL is None:
        logger.error("Prediction requested before model artifacts were created")
        return jsonify({"error": "Model not trained. Run: python scripts/generate_sample_data.py && python scripts/train_model.py"}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    target_date = payload.get("target_date")
    country = payload.get("country")
    if not isinstance(target_date, str) or not isinstance(country, str):
        return jsonify({"error": "Both target_date and country are required strings"}), 400
    try:
        parsed_date = date.fromisoformat(target_date)
    except ValueError:
        return jsonify({"error": "target_date must use YYYY-MM-DD format"}), 400
    if parsed_date.isoformat() != target_date:
        return jsonify({"error": "target_date must use YYYY-MM-DD format"}), 400

    if country.upper() != "ALL" and country not in COUNTRIES:
        return jsonify({"error": "Unknown country", "available_countries": COUNTRIES}), 400

    try:
        result = predict_revenue(MODEL, target_date, country, COUNTRIES)
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500

    logger.info("Prediction completed country=%s target_date=%s", country, target_date)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
