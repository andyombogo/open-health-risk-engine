"""
predict_risk.py
---------------
Inference utilities — load the trained model and score new individuals.

Used by:
  - dashboard/app.py (Streamlit UI)
  - External scripts / REST API (Phase 4 roadmap)

Example usage:
    from src.predict_risk import RiskPredictor

    predictor = RiskPredictor()
    result = predictor.predict({
        "age": 34,
        "sex_female": 1,
        "poverty_ratio": 1.8,
        "met_min_week": 150,
        "sleep_hours": 5.5,
        "sleep_trouble": 1,
        "bmi": 28.4,
        "drinks_per_week": 10,
        "education": 3,
    })
    print(result)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_FILENAME = "best_model.joblib"
FALLBACK_OPTIMAL_THRESHOLD = 0.35
OPTIMAL_THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.json"


def resolve_model_path(model_path: Union[str, Path, None] = None) -> Path:
    """Resolve the inference artifact path from an explicit value or env vars."""
    if model_path is not None:
        return Path(model_path)

    configured_path = os.getenv("OHRE_MODEL_PATH", "").strip()
    if configured_path:
        return Path(configured_path).expanduser()

    filename = os.getenv("OHRE_MODEL_FILENAME", DEFAULT_MODEL_FILENAME).strip()
    if not filename:
        filename = DEFAULT_MODEL_FILENAME
    return MODELS_DIR / filename


def load_optimal_threshold() -> float:
    """Load the tuned threshold from disk or return the requested fallback."""
    if OPTIMAL_THRESHOLD_PATH.exists():
        with open(OPTIMAL_THRESHOLD_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        threshold = float(payload["optimal_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "optimal_threshold in optimal_threshold.json must be between 0.0 and 1.0"
            )
        return threshold

    print(
        "optimal_threshold.json not found, using fallback 0.35 \u2014 retrain model to generate it."
    )
    return FALLBACK_OPTIMAL_THRESHOLD


def resolve_decision_threshold(decision_threshold: float | None = None) -> float:
    """Resolve and validate explicit threshold overrides for non-default callers."""
    if decision_threshold is None:
        env_threshold = os.getenv("OHRE_DECISION_THRESHOLD", "").strip()
        if env_threshold:
            decision_threshold = float(env_threshold)
        else:
            decision_threshold = load_optimal_threshold()

    threshold = float(decision_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0.0 and 1.0")
    return threshold


class RiskPredictor:
    """
    Wraps the trained Random Forest model with input validation,
    feature engineering, and interpretable output formatting.
    """

    SEVERITY_LABELS = {
        (0.0, 0.2): ("Minimal risk", "green"),
        (0.2, 0.4): ("Low risk", "blue"),
        (0.4, 0.6): ("Moderate risk", "orange"),
        (0.6, 0.8): ("High risk", "red"),
        (0.8, 1.0): ("Very high risk", "darkred"),
    }

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        decision_threshold: float | None = None,
    ):
        model_path = resolve_model_path(model_path)
        feat_path = MODELS_DIR / "feature_cols.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_model.py first."
            )
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Feature column artifact not found at {feat_path}. Run train_model.py first."
            )

        self.model_path = model_path
        self.threshold = resolve_decision_threshold(decision_threshold)
        self.decision_threshold = self.threshold
        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(feat_path)

    def _build_feature_row(self, inputs: dict) -> pd.DataFrame:
        """
        Convert raw user inputs into the full feature vector.
        Mirrors the logic in feature_engineering.py.
        """
        age = float(inputs.get("age", 35))
        sex_female = int(inputs.get("sex_female", 0))
        poverty_ratio = float(inputs.get("poverty_ratio", 2.5))
        met_min_week = float(inputs.get("met_min_week", 0))
        sleep_hours = float(inputs.get("sleep_hours", 7.0))
        sleep_trouble = int(inputs.get("sleep_trouble", 0))
        bmi = float(inputs.get("bmi", 25.0))
        drinks_per_week = float(inputs.get("drinks_per_week", 0))
        education = float(inputs.get("education", 3))
        race_eth = int(inputs.get("race_eth", 3))

        row = {
            # Demographics
            "age": age,
            "age_squared": age ** 2,
            "sex_female": sex_female,
            "poverty_ratio": poverty_ratio,
            "poverty_low": int(poverty_ratio < 1.0),
            "education": education,
            "race_mexican_american": int(race_eth == 1),
            "race_other_hispanic": int(race_eth == 2),
            "race_nh_black": int(race_eth == 4),
            "race_nh_asian": int(race_eth == 6),
            "race_other": int(race_eth == 7),
            # Physical activity
            "met_min_week": met_min_week,
            "met_log": np.log1p(met_min_week),
            "inactive": int(met_min_week == 0),
            "meets_who_guidelines": int(met_min_week >= 600),
            # Sleep
            "sleep_hours": sleep_hours,
            "sleep_trouble": sleep_trouble,
            "short_sleep": int(sleep_hours < 6),
            "long_sleep": int(sleep_hours > 9),
            "optimal_sleep": int(7 <= sleep_hours <= 9),
            # BMI
            "bmi": bmi,
            "underweight": int(bmi < 18.5),
            "overweight": int(bmi >= 25.0),
            "obese": int(bmi >= 30.0),
            # Alcohol
            "drinks_per_week": drinks_per_week,
            "drinks_log": np.log1p(drinks_per_week),
            "hazardous_drinking": int(
                drinks_per_week > (7 if sex_female == 1 else 14)
            ),
            # Interactions
            "inactive_x_poor_sleep": int(met_min_week == 0) * int(sleep_hours < 6),
            "poverty_x_inactive": int(poverty_ratio < 1.0) * int(met_min_week == 0),
            "age_x_poverty": age * int(poverty_ratio < 1.0),
        }

        return pd.DataFrame([row])[self.feature_cols]

    def get_severity_label(self, prob: float) -> tuple:
        """Map probability to a severity label and color."""
        for (low, high), (label, color) in self.SEVERITY_LABELS.items():
            if low <= prob < high:
                return label, color
        return "Very high risk", "darkred"

    def _get_explainability_model(self):
        """Return a fitted pipeline that exposes feature importances when available."""
        if hasattr(self.model, "named_steps"):
            return self.model

        estimator = getattr(self.model, "estimator", None)
        if estimator is not None and hasattr(estimator, "named_steps"):
            return estimator

        calibrated_classifiers = getattr(self.model, "calibrated_classifiers_", [])
        if calibrated_classifiers:
            fold_estimator = getattr(calibrated_classifiers[0], "estimator", None)
            if fold_estimator is not None and hasattr(fold_estimator, "named_steps"):
                return fold_estimator

        return None

    def predict(self, inputs: dict) -> dict:
        """
        Score a single individual and return structured output.

        Parameters
        ----------
        inputs : dict
            Keys: age, sex_female, poverty_ratio, met_min_week,
                  sleep_hours, sleep_trouble, bmi, drinks_per_week,
                  education, race_eth

        Returns
        -------
        dict with keys:
            risk_score    : float (0-1 probability)
            risk_label    : str (e.g. "Moderate risk")
            risk_color    : str (for UI display)
            phq9_estimate : float (rough PHQ-9 score estimate)
            decision_threshold : float (binary operating threshold)
            above_decision_threshold : bool
            top_factors   : list of dicts with feature name + direction
        """
        X = self._build_feature_row(inputs)
        prob = float(self.model.predict_proba(X)[0][1])
        label, color = self.get_severity_label(prob)
        # This threshold was calibrated on the precision-recall curve to maximize F1.
        binary_prediction = int(prob >= self.threshold)
        above_decision_threshold = bool(binary_prediction)

        # Rough PHQ-9 score estimate (linear scaling — not a clinical tool)
        phq9_estimate = round(prob * 27, 1)

        # Top risk factors from feature importances
        explainability_model = self._get_explainability_model()
        clf = (
            explainability_model.named_steps["clf"]
            if explainability_model is not None
            else None
        )
        if clf is not None and hasattr(clf, "feature_importances_"):
            importances = pd.Series(
                clf.feature_importances_, index=self.feature_cols
            )
            top_features = importances.nlargest(5)
            top_factors = [
                {
                    "feature": feat,
                    "importance": round(float(imp), 4),
                    "value": round(float(X[feat].values[0]), 3),
                }
                for feat, imp in top_features.items()
            ]
        else:
            top_factors = []

        return {
            "risk_score": round(prob, 4),
            "risk_label": label,
            "risk_color": color,
            "phq9_estimate": phq9_estimate,
            "decision_threshold": round(self.threshold, 4),
            "above_decision_threshold": above_decision_threshold,
            "top_factors": top_factors,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of individuals. Returns df with added risk columns."""
        results = []
        for _, row in df.iterrows():
            result = self.predict(row.to_dict())
            results.append(result)
        out = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), out], axis=1)


if __name__ == "__main__":
    # Demo: score a hypothetical individual
    predictor = RiskPredictor()

    test_cases = [
        {
            "name": "Low risk — active, good sleep",
            "age": 30, "sex_female": 0, "poverty_ratio": 3.5,
            "met_min_week": 900, "sleep_hours": 8.0, "sleep_trouble": 0,
            "bmi": 23.0, "drinks_per_week": 2, "education": 5, "race_eth": 3,
        },
        {
            "name": "High risk — inactive, poor sleep, poverty",
            "age": 45, "sex_female": 1, "poverty_ratio": 0.8,
            "met_min_week": 0, "sleep_hours": 5.0, "sleep_trouble": 1,
            "bmi": 32.0, "drinks_per_week": 14, "education": 2, "race_eth": 4,
        },
    ]

    for case in test_cases:
        name = case.pop("name")
        result = predictor.predict(case)
        print(f"\n{name}")
        print(f"  Risk score:    {result['risk_score']:.2%}")
        print(f"  Risk label:    {result['risk_label']}")
        print(f"  PHQ-9 estimate: {result['phq9_estimate']}")
        print(f"  Top factors:")
        for f in result["top_factors"]:
            print(f"    {f['feature']:30s} importance={f['importance']:.4f}")
