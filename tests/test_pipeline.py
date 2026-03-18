"""
tests/test_pipeline.py
-----------------------
Unit tests for data cleaning, feature engineering, and prediction.
Run with: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_cleaning import clean_phq9, clean_demographics, impute_missing
from src.feature_engineering import build_features, get_feature_columns


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_phq_data():
    """Minimal mock PHQ-9 DataFrame."""
    data = {
        "SEQN": [1, 2, 3, 4],
        "DPQ010": [0, 2, 3, 7],  # 7 = refused → NaN
        "DPQ020": [0, 2, 3, 1],
        "DPQ030": [1, 1, 2, 0],
        "DPQ040": [0, 3, 3, 0],
        "DPQ050": [0, 2, 3, 0],
        "DPQ060": [0, 1, 2, 0],
        "DPQ070": [0, 2, 3, 0],
        "DPQ080": [0, 1, 2, 0],
        "DPQ090": [0, 2, 3, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_demo_data():
    """Minimal mock demographics DataFrame."""
    data = {
        "SEQN": [1, 2, 3, 4, 5],
        "RIDAGEYR": [25, 16, 45, 60, 33],  # 16 should be excluded
        "RIAGENDR": [1, 2, 2, 1, 2],
        "RIDRETH3": [3, 3, 4, 1, 6],
        "DMDEDUC2": [4, 3, 5, 2, 3],
        "INDFMPIR": [2.5, 1.2, 0.8, 4.0, 6.0],  # 6.0 should be capped at 5.0
        "RIDEXPRG": [2, 1, 2, 2, 2],  # SEQN=2 is pregnant, should be excluded
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_features_df():
    """Minimal feature DataFrame for feature engineering tests."""
    return pd.DataFrame({
        "SEQN": [1, 2, 3],
        "age": [25, 45, 60],
        "sex_female": [0, 1, 1],
        "poverty_ratio": [0.8, 2.5, 4.0],
        "race_eth": [3, 4, 1],
        "education": [3, 4, 5],
        "met_min_week": [0, 600, 1200],
        "sleep_hours_avg": [5.0, 7.5, 8.0],
        "sleep_trouble": [1, 0, 0],
        "bmi": [22.0, 28.0, 32.0],
        "drinks_per_week_est": [0, 5, 15],
        "drinks_per_day": [0, 1, 3],
        "drink_frequency": [0, 52, 52],
        "phq9_score": [3.0, 8.0, 14.0],
        "depression_binary": [0, 0, 1],
    })


# ── PHQ-9 cleaning tests ──────────────────────────────────────────────────────

class TestCleanPHQ9:
    def test_phq9_score_computed(self, mock_phq_data):
        result = clean_phq9(mock_phq_data)
        assert "phq9_score" in result.columns

    def test_refused_code_becomes_nan(self, mock_phq_data):
        """SEQN=1 has DPQ010=7 (refused), but has 8 other valid items → should still compute."""
        result = clean_phq9(mock_phq_data)
        # SEQN 1: valid items = 8 (>= threshold of 8), score should be sum of non-refused
        row = result[result["SEQN"] == 1]
        assert row["phq9_score"].notna().all()

    def test_binary_outcome_correct(self, mock_phq_data):
        result = clean_phq9(mock_phq_data)
        # PHQ-9 >= 10 → depression_binary = 1
        for _, row in result.dropna(subset=["phq9_score"]).iterrows():
            expected = int(row["phq9_score"] >= 10)
            assert row["depression_binary"] == expected

    def test_severity_categories_assigned(self, mock_phq_data):
        result = clean_phq9(mock_phq_data)
        valid = result.dropna(subset=["phq9_severity"])
        assert all(valid["phq9_severity"].isin(
            ["minimal", "mild", "moderate", "moderately_severe", "severe"]
        ))

    def test_output_columns(self, mock_phq_data):
        result = clean_phq9(mock_phq_data)
        assert set(result.columns) == {"SEQN", "phq9_score", "phq9_severity", "depression_binary"}


# ── Demographics cleaning tests ───────────────────────────────────────────────

class TestCleanDemographics:
    def test_minors_excluded(self, mock_demo_data):
        result = clean_demographics(mock_demo_data)
        assert all(result["age"] >= 18)

    def test_pregnant_excluded(self, mock_demo_data):
        result = clean_demographics(mock_demo_data)
        # SEQN=2 was pregnant and should be excluded
        assert 2 not in result["SEQN"].values

    def test_poverty_capped(self, mock_demo_data):
        result = clean_demographics(mock_demo_data)
        assert result["poverty_ratio"].max() <= 5.0

    def test_sex_encoded(self, mock_demo_data):
        result = clean_demographics(mock_demo_data)
        assert set(result["sex_female"].unique()).issubset({0, 1})

    def test_female_encoding_correct(self, mock_demo_data):
        result = clean_demographics(mock_demo_data)
        # SEQN=3 is RIAGENDR=2 (female) → sex_female=1
        row = result[result["SEQN"] == 3]
        if len(row) > 0:
            assert row["sex_female"].values[0] == 1


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestBuildFeatures:
    def test_output_contains_all_feature_columns(self, mock_features_df):
        result = build_features(mock_features_df)
        expected_cols = get_feature_columns()
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_activity_flags_correct(self, mock_features_df):
        result = build_features(mock_features_df)
        # Row 0: met_min_week=0 → inactive=1, meets_who_guidelines=0
        assert result.iloc[0]["inactive"] == 1
        assert result.iloc[0]["meets_who_guidelines"] == 0
        # Row 1: met_min_week=600 → inactive=0, meets_who_guidelines=1
        assert result.iloc[1]["inactive"] == 0
        assert result.iloc[1]["meets_who_guidelines"] == 1

    def test_sleep_flags_correct(self, mock_features_df):
        result = build_features(mock_features_df)
        # Row 0: sleep=5.0 → short_sleep=1, optimal_sleep=0
        assert result.iloc[0]["short_sleep"] == 1
        assert result.iloc[0]["optimal_sleep"] == 0

    def test_interaction_term(self, mock_features_df):
        result = build_features(mock_features_df)
        # Row 0: inactive=1, short_sleep=1 → interaction=1
        assert result.iloc[0]["inactive_x_poor_sleep"] == 1

    def test_hazardous_drinking_sex_specific(self, mock_features_df):
        result = build_features(mock_features_df)
        # Row 2: female, drinks=15/wk → >7 threshold → hazardous_drinking=1
        assert result.iloc[2]["hazardous_drinking"] == 1

    def test_no_nulls_in_feature_columns(self, mock_features_df):
        result = build_features(mock_features_df)
        feature_cols = get_feature_columns()
        null_counts = result[feature_cols].isnull().sum()
        assert null_counts.sum() == 0, f"Null values found: {null_counts[null_counts > 0]}"

    def test_age_squared_positive(self, mock_features_df):
        result = build_features(mock_features_df)
        assert (result["age_squared"] >= 0).all()


# ── Imputation tests ──────────────────────────────────────────────────────────

class TestImputation:
    def test_no_nulls_after_imputation(self):
        df = pd.DataFrame({
            "SEQN": [1, 2, 3],
            "age": [25.0, np.nan, 50.0],
            "bmi": [22.0, 28.0, np.nan],
            "poverty_ratio": [np.nan, 2.0, 3.0],
            "education": [3, np.nan, 4],
            "met_min_week": [300, np.nan, 600],
            "sleep_hours_avg": [7.0, 6.0, np.nan],
            "drinks_per_day": [np.nan, 2.0, 1.0],
            "drinks_per_week_est": [0.0, np.nan, 5.0],
            "drink_frequency": [np.nan, 12.0, 24.0],
            "race_eth": [3, np.nan, 4],
        })
        result = impute_missing(df)
        for col in ["age", "bmi", "poverty_ratio", "met_min_week",
                    "sleep_hours_avg", "drinks_per_day", "drinks_per_week_est",
                    "drink_frequency"]:
            if col in result.columns:
                assert result[col].isnull().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
