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
# Backward-compatible alias used by older validation utilities.
DEFAULT_DECISION_THRESHOLD = FALLBACK_OPTIMAL_THRESHOLD
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
    Wraps the trained deployment model with input validation,
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
        marital_status = int(inputs.get("marital_status", 1))
        born_us = int(inputs.get("born_us", 1))
        met_min_week = float(inputs.get("met_min_week", 0))
        sedentary_minutes = float(inputs.get("sedentary_minutes", 360))
        sleep_hours = float(inputs.get("sleep_hours", 7.0))
        sleep_trouble = int(inputs.get("sleep_trouble", 0))
        sleep_apnea_symptom_freq = float(inputs.get("sleep_apnea_symptom_freq", 0))
        daytime_sleepiness_freq = float(inputs.get("daytime_sleepiness_freq", 1))
        bmi = float(inputs.get("bmi", 25.0))
        drinks_per_week = float(inputs.get("drinks_per_week", 0))
        ever_smoked_100_cigs = int(inputs.get("ever_smoked_100_cigs", 0))
        current_smoking_status = float(inputs.get("current_smoking_status", 3))
        days_smoked_past_month = float(inputs.get("days_smoked_past_month", 0))
        cigs_per_day_smoking_days = float(inputs.get("cigs_per_day_smoking_days", 0))
        quit_attempt_last_year = int(inputs.get("quit_attempt_last_year", 0))
        general_health = float(inputs.get("general_health", 3))
        healthcare_visits_code = float(inputs.get("healthcare_visits_code", 2))
        hospitalized_last_year = int(inputs.get("hospitalized_last_year", 0))
        routine_care_place = float(inputs.get("routine_care_place", 1))
        insurance_gap_last_year = int(inputs.get("insurance_gap_last_year", 0))
        insured = int(inputs.get("insured", 1))
        education = float(inputs.get("education", 3))
        race_eth = int(inputs.get("race_eth", 3))
        poverty_low = int(poverty_ratio < 1.0)
        inactive = int(met_min_week == 0)
        sleep_trouble_flag = int(sleep_trouble)
        short_sleep = int(sleep_hours < 6)
        long_sleep = int(sleep_hours > 9)
        optimal_sleep = int(7 <= sleep_hours <= 9)
        underweight = int(bmi < 18.5)
        overweight = int(bmi >= 25.0)
        obese = int(bmi >= 30.0)
        hazardous_drinking = int(drinks_per_week > (7 if sex_female == 1 else 14))
        lifestyle_burden_count = (
            inactive
            + short_sleep
            + sleep_trouble_flag
            + hazardous_drinking
            + obese
        )

        row = {
            # Demographics
            "age": age,
            "age_squared": age ** 2,
            "age_18_34": int(18 <= age < 35),
            "age_35_49": int(35 <= age < 50),
            "age_50_64": int(50 <= age < 65),
            "age_65_plus": int(age >= 65),
            "sex_female": sex_female,
            "poverty_ratio": poverty_ratio,
            "poverty_low": poverty_low,
            "education": education,
            "marital_formerly_married": int(marital_status in {2, 3, 4}),
            "marital_never_married": int(marital_status == 5),
            "foreign_born": int(born_us == 2),
            "race_mexican_american": int(race_eth == 1),
            "race_other_hispanic": int(race_eth == 2),
            "race_nh_black": int(race_eth == 4),
            "race_nh_asian": int(race_eth == 6),
            "race_other": int(race_eth == 7),
            # Physical activity
            "met_min_week": met_min_week,
            "met_log": np.log1p(met_min_week),
            "inactive": inactive,
            "meets_who_guidelines": int(met_min_week >= 600),
            "sedentary_minutes": sedentary_minutes,
            "sedentary_high": int(sedentary_minutes >= 480),
            # Sleep
            "sleep_hours": sleep_hours,
            "sleep_trouble": sleep_trouble_flag,
            "short_sleep": short_sleep,
            "long_sleep": long_sleep,
            "optimal_sleep": optimal_sleep,
            "sleep_apnea_symptom_freq": sleep_apnea_symptom_freq,
            "sleep_apnea_symptom_high": int(sleep_apnea_symptom_freq >= 2),
            "daytime_sleepiness_freq": daytime_sleepiness_freq,
            "daytime_sleepy_often": int(daytime_sleepiness_freq >= 3),
            # BMI
            "bmi": bmi,
            "underweight": underweight,
            "overweight": overweight,
            "obese": obese,
            # Alcohol
            "drinks_per_week": drinks_per_week,
            "drinks_log": np.log1p(drinks_per_week),
            "hazardous_drinking": hazardous_drinking,
            # Smoking
            "ever_smoked_100_cigs": ever_smoked_100_cigs,
            "current_smoker": int(current_smoking_status in {1, 2}),
            "daily_smoker": int(current_smoking_status == 1),
            "days_smoked_past_month": days_smoked_past_month,
            "cigs_per_day_smoking_days": cigs_per_day_smoking_days,
            "heavy_smoker": int(cigs_per_day_smoking_days >= 10),
            "quit_attempt_last_year": quit_attempt_last_year,
            # General health and healthcare access
            "general_health_score": general_health,
            "fair_poor_health": int(general_health >= 4),
            "high_healthcare_use": int(healthcare_visits_code >= 4),
            "hospitalized_last_year": hospitalized_last_year,
            "routine_care_absent": int(routine_care_place != 1),
            "insurance_gap_last_year": insurance_gap_last_year,
            "insured": insured,
            # Composite scores
            "lifestyle_burden_count": lifestyle_burden_count,
            "high_burden": int(lifestyle_burden_count >= 3),
            "social_vuln_score": poverty_low + int(education <= 2) + int(age >= 60),
            "phys_health_risk_score": obese + int(bmi >= 35) + underweight + inactive,
            # Interactions
            "inactive_x_poor_sleep": inactive * short_sleep,
            "sleep_activity_risk": sleep_trouble_flag * inactive,
            "poverty_x_inactive": poverty_low * inactive,
            "poverty_x_sleep_trouble": poverty_low * sleep_trouble_flag,
            "age_x_poverty": age * poverty_low,
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

    def _get_model_feature_names(self) -> list:
        """Return feature names aligned to the fitted estimator after selection."""
        explainability_model = self._get_explainability_model()
        feature_names = np.array(self.feature_cols)
        if explainability_model is None:
            return feature_names.tolist()

        selector = explainability_model.named_steps.get("selector")
        if selector is not None:
            feature_names = feature_names[selector.get_support()]
        return feature_names.tolist()

    def _get_top_factors(self, X: pd.DataFrame) -> list:
        """Return top per-prediction factors for tree or linear deployment models."""
        explainability_model = self._get_explainability_model()
        clf = (
            explainability_model.named_steps["clf"]
            if explainability_model is not None
            else None
        )
        if clf is None:
            return []

        model_feature_names = self._get_model_feature_names()

        if hasattr(clf, "feature_importances_"):
            importances = pd.Series(
                clf.feature_importances_, index=model_feature_names
            )
            top_features = importances.nlargest(5)
            return [
                {
                    "feature": feat,
                    "importance": round(float(imp), 4),
                    "value": round(float(X[feat].values[0]), 3),
                }
                for feat, imp in top_features.items()
            ]

        if hasattr(clf, "coef_"):
            transformed_X = X
            scaler = explainability_model.named_steps.get("scaler")
            if scaler is not None:
                transformed_X = scaler.transform(X)

            contributions = clf.coef_.ravel() * np.asarray(transformed_X)[0]
            contribution_series = pd.Series(contributions, index=model_feature_names)
            top_features = contribution_series.abs().nlargest(5).index
            return [
                {
                    "feature": feat,
                    "importance": round(float(abs(contribution_series[feat])), 4),
                    "value": round(float(X[feat].values[0]), 3),
                    "direction": (
                        "increase_risk"
                        if contribution_series[feat] >= 0
                        else "decrease_risk"
                    ),
                }
                for feat in top_features
            ]

        return []

    def predict(self, inputs: dict) -> dict:
        """
        Score a single individual and return structured output.

        Parameters
        ----------
        inputs : dict
            Keys: age, sex_female, poverty_ratio, met_min_week,
                  sleep_hours, sleep_trouble, bmi, drinks_per_week,
                  education, race_eth. Optional extended keys include
                  marital_status, born_us, sedentary_minutes,
                  sleep_apnea_symptom_freq, daytime_sleepiness_freq,
                  ever_smoked_100_cigs, current_smoking_status,
                  days_smoked_past_month, cigs_per_day_smoking_days,
                  quit_attempt_last_year, general_health,
                  healthcare_visits_code, hospitalized_last_year,
                  routine_care_place, insurance_gap_last_year, insured.

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
        # This threshold was calibrated on the precision-recall curve to maximize
        # F1 while preserving at least 0.70 recall for screening use.
        binary_prediction = int(prob >= self.threshold)
        above_decision_threshold = bool(binary_prediction)

        # Rough PHQ-9 score estimate (linear scaling — not a clinical tool)
        phq9_estimate = round(prob * 27, 1)

        top_factors = self._get_top_factors(X)

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
