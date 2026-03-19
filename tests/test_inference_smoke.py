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


def test_risk_predictor_encodes_all_supported_race_groups():
    predictor = RiskPredictor()
    race_feature_map = {
        1: "race_mexican_american",
        2: "race_other_hispanic",
        3: None,
        4: "race_nh_black",
        6: "race_nh_asian",
        7: "race_other",
    }
    race_features = {
        "race_mexican_american",
        "race_other_hispanic",
        "race_nh_black",
        "race_nh_asian",
        "race_other",
    }
    base_inputs = {
        "age": 35,
        "sex_female": 0,
        "poverty_ratio": 2.5,
        "met_min_week": 300,
        "sleep_hours": 7.0,
        "sleep_trouble": 0,
        "bmi": 24.0,
        "drinks_per_week": 3,
        "education": 4,
    }

    for race_eth, active_feature in race_feature_map.items():
        X = predictor._build_feature_row({**base_inputs, "race_eth": race_eth})
        active_features = {
            feature for feature in race_features if int(X.loc[0, feature]) == 1
        }
        expected = {active_feature} if active_feature else set()
        assert active_features == expected
