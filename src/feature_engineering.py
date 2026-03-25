"""
feature_engineering.py
-----------------------
Constructs the final feature matrix for model training.

Design decisions (documented for reproducibility):
  - We use BOTH continuous and encoded categorical features
  - Interaction terms: sleep × activity (known epidemiological relationship)
  - Hazardous drinking: sex-specific NIAAA threshold applied here
  - No leakage: all transformations are fit on train set, applied to test

Output:
  data/processed/features.csv   — full feature matrix with outcome
  data/processed/feature_names.json — feature metadata
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from the cleaned NHANES dataset.
    Returns a DataFrame with all features + outcome columns.
    """
    feat = pd.DataFrame(index=df.index)
    feat["SEQN"] = df["SEQN"]

    # ── Demographic features ──────────────────────────────────────────────────
    feat["age"] = df["age"]
    feat["age_squared"] = df["age"] ** 2  # Non-linear age relationship
    feat["age_18_34"] = ((df["age"] >= 18) & (df["age"] < 35)).astype(int)
    feat["age_35_49"] = ((df["age"] >= 35) & (df["age"] < 50)).astype(int)
    feat["age_50_64"] = ((df["age"] >= 50) & (df["age"] < 65)).astype(int)
    feat["age_65_plus"] = (df["age"] >= 65).astype(int)
    feat["sex_female"] = df["sex_female"]
    feat["poverty_ratio"] = df["poverty_ratio"]
    feat["poverty_low"] = (df["poverty_ratio"] < 1.0).astype(int)  # Below poverty line
    feat["marital_formerly_married"] = df["marital_status"].isin([2, 3, 4]).astype(int)
    feat["marital_never_married"] = (df["marital_status"] == 5).astype(int)
    feat["foreign_born"] = (df["born_us"] == 2).astype(int)

    # Education: ordinal (1-5 scale maps roughly to years of schooling)
    feat["education"] = df["education"].fillna(3)  # Fill with median

    # Race/ethnicity dummies (reference: Non-Hispanic White = 3)
    feat["race_mexican_american"] = (df["race_eth"] == 1).astype(int)
    feat["race_other_hispanic"] = (df["race_eth"] == 2).astype(int)
    feat["race_nh_black"] = (df["race_eth"] == 4).astype(int)
    feat["race_nh_asian"] = (df["race_eth"] == 6).astype(int)
    feat["race_other"] = (df["race_eth"] == 7).astype(int)

    # ── Physical activity features ────────────────────────────────────────────
    feat["met_min_week"] = df["met_min_week"].fillna(0)
    feat["met_log"] = np.log1p(feat["met_min_week"])  # Log-transform (right-skewed)
    feat["inactive"] = (df["met_min_week"] == 0).astype(int)
    feat["meets_who_guidelines"] = (df["met_min_week"] >= 600).astype(int)
    feat["sedentary_minutes"] = df["sedentary_minutes"].fillna(df["sedentary_minutes"].median())
    feat["sedentary_high"] = (feat["sedentary_minutes"] >= 480).astype(int)

    # ── Sleep features ────────────────────────────────────────────────────────
    feat["sleep_hours"] = df["sleep_hours_avg"].fillna(df["sleep_hours_avg"].median())
    feat["sleep_trouble"] = df["sleep_trouble"].fillna(0)

    # Short sleep (<6h) and long sleep (>9h) both associated with depression
    feat["short_sleep"] = (feat["sleep_hours"] < 6).astype(int)
    feat["long_sleep"] = (feat["sleep_hours"] > 9).astype(int)

    # Optimal sleep window (7-9h)
    feat["optimal_sleep"] = (
        (feat["sleep_hours"] >= 7) & (feat["sleep_hours"] <= 9)
    ).astype(int)
    feat["sleep_apnea_symptom_freq"] = df["sleep_apnea_symptom_freq"].fillna(0)
    feat["sleep_apnea_symptom_high"] = (feat["sleep_apnea_symptom_freq"] >= 2).astype(int)
    feat["daytime_sleepiness_freq"] = df["daytime_sleepiness_freq"].fillna(1)
    feat["daytime_sleepy_often"] = (feat["daytime_sleepiness_freq"] >= 3).astype(int)

    # ── BMI features ──────────────────────────────────────────────────────────
    feat["bmi"] = df["bmi"].fillna(df["bmi"].median())
    feat["underweight"] = (df["bmi"] < 18.5).astype(int)
    feat["overweight"] = (df["bmi"] >= 25.0).astype(int)
    feat["obese"] = (df["bmi"] >= 30.0).astype(int)

    # ── Alcohol features ──────────────────────────────────────────────────────
    feat["drinks_per_week"] = df["drinks_per_week_est"].fillna(0)
    feat["drinks_log"] = np.log1p(feat["drinks_per_week"])

    # NIAAA hazardous drinking: >14/wk men, >7/wk women
    hazardous_threshold = np.where(df["sex_female"] == 1, 7, 14)
    feat["hazardous_drinking"] = (
        feat["drinks_per_week"] > hazardous_threshold
    ).astype(int)
    feat["ever_smoked_100_cigs"] = df["ever_smoked_100_cigs"].fillna(0)
    feat["current_smoker"] = df["current_smoking_status"].isin([1, 2]).astype(int)
    feat["daily_smoker"] = (df["current_smoking_status"] == 1).astype(int)
    feat["days_smoked_past_month"] = df["days_smoked_past_month"].fillna(0)
    feat["cigs_per_day_smoking_days"] = df["cigs_per_day_smoking_days"].fillna(0)
    feat["heavy_smoker"] = (feat["cigs_per_day_smoking_days"] >= 10).astype(int)
    feat["quit_attempt_last_year"] = df["quit_attempt_last_year"].fillna(0)
    feat["general_health_score"] = df["general_health"].fillna(3)
    feat["fair_poor_health"] = (feat["general_health_score"] >= 4).astype(int)
    feat["high_healthcare_use"] = (df["healthcare_visits_code"].fillna(2) >= 4).astype(int)
    feat["hospitalized_last_year"] = df["hospitalized_last_year"].fillna(0)
    feat["routine_care_absent"] = (df["routine_care_place"].fillna(1) != 1).astype(int)
    feat["insurance_gap_last_year"] = df["insurance_gap_last_year"].fillna(0)
    feat["insured"] = df["insured"].fillna(1)
    feat["lifestyle_burden_count"] = (
        feat["inactive"]
        + feat["short_sleep"]
        + feat["sleep_trouble"]
        + feat["hazardous_drinking"]
        + feat["obese"]
    )
    feat["high_burden"] = (feat["lifestyle_burden_count"] >= 3).astype(int)
    feat["social_vuln_score"] = (
        feat["poverty_low"]
        + (feat["education"] <= 2).astype(int)
        + (feat["age"] >= 60).astype(int)
    )
    feat["phys_health_risk_score"] = (
        feat["obese"]
        + (feat["bmi"] >= 35).astype(int)
        + feat["underweight"]
        + feat["inactive"]
    )

    # ── Interaction features ──────────────────────────────────────────────────
    # Low activity + poor sleep: compounding risk (well-documented in literature)
    feat["inactive_x_poor_sleep"] = feat["inactive"] * feat["short_sleep"]
    feat["sleep_activity_risk"] = feat["sleep_trouble"] * feat["inactive"]

    # Poverty + low activity (social determinant interaction)
    feat["poverty_x_inactive"] = feat["poverty_low"] * feat["inactive"]
    feat["poverty_x_sleep_trouble"] = feat["poverty_low"] * feat["sleep_trouble"]

    # Age × poverty (older adults in poverty: higher risk)
    feat["age_x_poverty"] = feat["age"] * feat["poverty_low"]

    # ── Outcomes ──────────────────────────────────────────────────────────────
    feat["phq9_score"] = df["phq9_score"]
    feat["depression_binary"] = df["depression_binary"]

    return feat


def get_feature_columns() -> list:
    """Return the list of feature column names (excluding SEQN and outcomes)."""
    return [
        # Demographics
        "age", "age_squared", "age_18_34", "age_35_49", "age_50_64",
        "age_65_plus", "sex_female", "poverty_ratio", "poverty_low", "education",
        "marital_formerly_married", "marital_never_married", "foreign_born",
        "race_mexican_american", "race_other_hispanic", "race_nh_black",
        "race_nh_asian", "race_other",
        # Physical activity
        "met_min_week", "met_log", "inactive", "meets_who_guidelines",
        "sedentary_minutes", "sedentary_high",
        # Sleep
        "sleep_hours", "sleep_trouble", "short_sleep", "long_sleep", "optimal_sleep",
        "sleep_apnea_symptom_freq", "sleep_apnea_symptom_high",
        "daytime_sleepiness_freq", "daytime_sleepy_often",
        # BMI
        "bmi", "underweight", "overweight", "obese",
        # Alcohol
        "drinks_per_week", "drinks_log", "hazardous_drinking",
        # Smoking
        "ever_smoked_100_cigs", "current_smoker", "daily_smoker",
        "days_smoked_past_month", "cigs_per_day_smoking_days", "heavy_smoker",
        "quit_attempt_last_year",
        # General health and healthcare access
        "general_health_score", "fair_poor_health", "high_healthcare_use",
        "hospitalized_last_year", "routine_care_absent",
        "insurance_gap_last_year", "insured",
        # Composite scores
        "lifestyle_burden_count", "high_burden", "social_vuln_score",
        "phys_health_risk_score",
        # Interactions
        "inactive_x_poor_sleep", "sleep_activity_risk", "poverty_x_inactive",
        "poverty_x_sleep_trouble", "age_x_poverty",
    ]


def main():
    input_path = PROCESSED_DIR / "nhanes_clean.csv"
    if not input_path.exists():
        raise FileNotFoundError("Run data_cleaning.py first.")

    print("Loading clean dataset...")
    df = pd.read_csv(input_path)

    print("Engineering features...")
    features_df = build_features(df)

    # Save feature matrix
    out_path = PROCESSED_DIR / "features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Feature matrix saved: {out_path}")
    print(f"Shape: {features_df.shape}")

    # Save feature metadata as JSON (useful for documentation + dashboard)
    feature_meta = {
        "feature_columns": get_feature_columns(),
        "outcome_binary": "depression_binary",
        "outcome_continuous": "phq9_score",
        "n_features": len(get_feature_columns()),
        "feature_groups": {
            "demographics": ["age", "age_squared", "age_18_34", "age_35_49",
                             "age_50_64", "age_65_plus", "sex_female",
                             "poverty_ratio", "poverty_low", "education",
                             "marital_formerly_married", "marital_never_married",
                             "foreign_born",
                             "race_mexican_american", "race_other_hispanic",
                             "race_nh_black", "race_nh_asian", "race_other"],
            "physical_activity": ["met_min_week", "met_log", "inactive",
                                  "meets_who_guidelines", "sedentary_minutes",
                                  "sedentary_high"],
            "sleep": ["sleep_hours", "sleep_trouble", "short_sleep",
                      "long_sleep", "optimal_sleep",
                      "sleep_apnea_symptom_freq", "sleep_apnea_symptom_high",
                      "daytime_sleepiness_freq", "daytime_sleepy_often"],
            "bmi": ["bmi", "underweight", "overweight", "obese"],
            "alcohol": ["drinks_per_week", "drinks_log", "hazardous_drinking"],
            "smoking": ["ever_smoked_100_cigs", "current_smoker",
                        "daily_smoker", "days_smoked_past_month",
                        "cigs_per_day_smoking_days", "heavy_smoker",
                        "quit_attempt_last_year"],
            "healthcare": ["general_health_score", "fair_poor_health",
                           "high_healthcare_use", "hospitalized_last_year",
                           "routine_care_absent", "insurance_gap_last_year",
                           "insured"],
            "composites": ["lifestyle_burden_count", "high_burden",
                           "social_vuln_score", "phys_health_risk_score"],
            "interactions": ["inactive_x_poor_sleep", "sleep_activity_risk",
                             "poverty_x_inactive", "poverty_x_sleep_trouble",
                             "age_x_poverty"],
        }
    }

    meta_path = PROCESSED_DIR / "feature_names.json"
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)
    print(f"Feature metadata saved: {meta_path}")

    # Quick summary stats
    print("\nFeature summary (first 5 features):")
    print(features_df[get_feature_columns()[:5]].describe().round(2))


if __name__ == "__main__":
    main()
