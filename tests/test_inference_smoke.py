import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict_risk import RiskPredictor


def test_risk_predictor_smoke():
    predictor = RiskPredictor()

    result = predictor.predict(
        {
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
    )

    assert 0.0 <= result["risk_score"] <= 1.0
    assert isinstance(result["risk_label"], str) and result["risk_label"]
    assert isinstance(result["risk_color"], str) and result["risk_color"]
    assert 0.0 <= result["phq9_estimate"] <= 27.0
    assert isinstance(result["top_factors"], list)
    assert len(result["top_factors"]) > 0
