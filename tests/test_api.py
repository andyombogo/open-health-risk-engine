import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api import create_app
from src.predict_risk import resolve_decision_threshold, resolve_model_path


def make_client(rate_limit_per_minute: int = 5) -> TestClient:
    app = create_app(
        api_key="test-key",
        rate_limit_per_minute=rate_limit_per_minute,
        request_log_path=None,
    )
    return TestClient(app)


def sample_payload() -> dict:
    return {
        "age": 35,
        "sex_female": 0,
        "poverty_ratio": 2.5,
        "met_min_week": 300,
        "sleep_hours": 7.0,
        "sleep_trouble": 0,
        "bmi": 24.0,
        "drinks_per_week": 3,
        "education": 4,
        "race_eth": 3,
    }


def test_health_endpoint_reports_ready_state():
    client = make_client()
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["auth_enabled"] is True
    assert payload["missing_artifacts"] == []
    expected_model_name = resolve_model_path().name
    expected_threshold = round(resolve_decision_threshold(), 4)
    assert payload["model_artifact"] == expected_model_name
    assert payload["decision_threshold"] == expected_threshold
    assert payload["calibrated_scores"] == ("calibrated" in expected_model_name)


def test_predict_requires_api_key():
    client = make_client()
    response = client.post("/predict", json=sample_payload())

    assert response.status_code == 401
    assert "X-API-Key" in response.json()["detail"]


def test_predict_returns_structured_result():
    client = make_client()
    response = client.post(
        "/predict",
        headers={"X-API-Key": "test-key"},
        json=sample_payload(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert payload["risk_label"]
    assert 0.0 <= payload["phq9_estimate"] <= 27.0
    assert 0.0 <= payload["decision_threshold"] <= 1.0
    assert payload["above_decision_threshold"] == (
        payload["risk_score"] >= payload["decision_threshold"]
    )
    assert isinstance(payload["top_factors"], list)


def test_predict_validates_schema():
    client = make_client()
    invalid_payload = sample_payload()
    invalid_payload["race_eth"] = 5

    response = client.post(
        "/predict",
        headers={"X-API-Key": "test-key"},
        json=invalid_payload,
    )

    assert response.status_code == 422
    assert "race_eth" in response.text


def test_predict_rate_limits_repeated_calls():
    client = make_client(rate_limit_per_minute=2)
    headers = {"X-API-Key": "test-key"}

    first = client.post("/predict", headers=headers, json=sample_payload())
    second = client.post("/predict", headers=headers, json=sample_payload())
    third = client.post("/predict", headers=headers, json=sample_payload())

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
