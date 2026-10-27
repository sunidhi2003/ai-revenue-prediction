import os
import subprocess
import sys
import pytest

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    os.makedirs("data", exist_ok=True)
    subprocess.run([sys.executable, "scripts/generate_sample_data.py"], check=True)
    subprocess.run([sys.executable, "scripts/train_model.py"], check=True)

def test_health_endpoint():
    from app import app
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"

def test_predict_one_country():
    from app import app
    client = app.test_client()
    response = client.post("/predict", json={"target_date": "2026-01-10", "country": "Canada"})
    assert response.status_code == 200
    assert response.json["predicted_revenue"] >= 0

def test_predict_all_countries():
    from app import app
    client = app.test_client()
    response = client.post("/predict", json={"target_date": "2026-01-10", "country": "ALL"})
    assert response.status_code == 200
    assert response.json["country"] == "ALL"
    assert len(response.json["breakdown"]) > 0

def test_invalid_input():
    from app import app
    client = app.test_client()
    response = client.post("/predict", json={"target_date": "tomorrow", "country": "Canada"})
    assert response.status_code == 400
