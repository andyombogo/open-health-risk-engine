"""
error_analysis.py
-----------------
Generate false-positive and false-negative analysis artifacts for the deployed
 model using the same held-out split definition as train_model.py.

Outputs:
  models/error_outcome_summary.csv
  models/error_feature_deltas.csv
  models/error_rate_by_subgroup.csv
  figures/confusion_matrix_random_forest.png
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / Multiracial",
}

USER_FACING_FEATURES = [
    "age",
    "sex_female",
    "poverty_ratio",
    "met_min_week",
    "sleep_hours_avg",
    "sleep_trouble",
    "bmi",
    "education",
    "phq9_score",
    "y_prob",
]


def load_error_frame():
    """Load the fitted model inputs and cleaned explanatory fields."""
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    nhanes = pd.read_csv(
        PROCESSED_DIR / "nhanes_clean.csv",
        usecols=[
            "SEQN",
            "age",
            "sex_female",
            "race_eth",
            "education",
            "poverty_ratio",
            "met_min_week",
            "sleep_hours_avg",
            "sleep_trouble",
            "bmi",
            "phq9_score",
            "depression_binary",
        ],
    )
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    model = joblib.load(MODELS_DIR / "best_model.joblib")

    frame = features.merge(
        nhanes,
        on="SEQN",
        how="left",
        suffixes=("_feat", ""),
    )

    X = frame[feature_cols]
    y = frame["depression_binary"].astype(int)
    _, X_test, _, y_test, _, frame_test = train_test_split(
        X,
        y,
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    frame_test = frame_test.copy()
    frame_test["y_true"] = y_test.to_numpy()
    frame_test["y_prob"] = model.predict_proba(X_test)[:, 1]
    frame_test["y_pred"] = (frame_test["y_prob"] >= 0.5).astype(int)

    frame_test["error_outcome"] = np.select(
        [
            (frame_test["y_true"] == 1) & (frame_test["y_pred"] == 1),
            (frame_test["y_true"] == 0) & (frame_test["y_pred"] == 1),
            (frame_test["y_true"] == 1) & (frame_test["y_pred"] == 0),
            (frame_test["y_true"] == 0) & (frame_test["y_pred"] == 0),
        ],
        ["TP", "FP", "FN", "TN"],
        default="other",
    )

    frame_test["sex_group"] = np.where(
        frame_test["sex_female"] == 1, "Female", "Male"
    )
    frame_test["age_band"] = pd.cut(
        frame_test["age"],
        bins=[18, 35, 50, 65, np.inf],
        right=False,
        labels=["18-34", "35-49", "50-64", "65+"],
    ).astype(str)
    frame_test["poverty_band"] = pd.cut(
        frame_test["poverty_ratio"],
        bins=[-np.inf, 1.0, 2.0, np.inf],
        right=False,
        labels=["<1.0", "1.0-1.99", "2.0+"],
    ).astype(str)
    frame_test["race_group"] = frame_test["race_eth"].map(RACE_LABELS).fillna("Unknown")

    return frame_test


def make_outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the four confusion outcomes using user-facing variables."""
    summary = (
        frame.groupby("error_outcome", observed=False)
        .agg(
            n=("SEQN", "size"),
            share_of_test=("SEQN", lambda s: len(s) / len(frame)),
            mean_predicted_probability=("y_prob", "mean"),
            mean_phq9_score=("phq9_score", "mean"),
            female_share=("sex_female", "mean"),
            mean_age=("age", "mean"),
            mean_poverty_ratio=("poverty_ratio", "mean"),
            mean_met_min_week=("met_min_week", "mean"),
            mean_sleep_hours=("sleep_hours_avg", "mean"),
            sleep_trouble_rate=("sleep_trouble", "mean"),
            mean_bmi=("bmi", "mean"),
            mean_education=("education", "mean"),
        )
        .reset_index()
    )
    return summary.round(4)


def make_feature_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare FP vs TN and FN vs TP for interpretable pattern differences."""
    comparisons = [("FP", "TN"), ("FN", "TP")]
    labels = {
        "age": "Age",
        "sex_female": "Female share",
        "poverty_ratio": "Poverty ratio",
        "met_min_week": "Weekly activity (MET-min)",
        "sleep_hours_avg": "Sleep hours",
        "sleep_trouble": "Sleep trouble rate",
        "bmi": "BMI",
        "education": "Education",
        "phq9_score": "PHQ-9 score",
        "y_prob": "Predicted probability",
    }

    rows = []
    for left, right in comparisons:
        left_df = frame[frame["error_outcome"] == left]
        right_df = frame[frame["error_outcome"] == right]
        for feature in USER_FACING_FEATURES:
            left_value = float(left_df[feature].mean())
            right_value = float(right_df[feature].mean())
            rows.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "feature": feature,
                    "feature_label": labels[feature],
                    "left_group_mean": round(left_value, 4),
                    "right_group_mean": round(right_value, 4),
                    "delta_left_minus_right": round(left_value - right_value, 4),
                }
            )
    return pd.DataFrame(rows)


def make_error_rate_by_subgroup(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute subgroup-level false positive and false negative rates."""
    rows = []
    subgroup_specs = [
        ("sex", "sex_group"),
        ("age_band", "age_band"),
        ("poverty_band", "poverty_band"),
        ("race", "race_group"),
    ]

    for subgroup_type, column in subgroup_specs:
        for subgroup_value, group_df in frame.groupby(column, observed=False):
            negatives = group_df[group_df["y_true"] == 0]
            positives = group_df[group_df["y_true"] == 1]
            false_positive_rate = (
                float((negatives["y_pred"] == 1).mean()) if len(negatives) else np.nan
            )
            false_negative_rate = (
                float((positives["y_pred"] == 0).mean()) if len(positives) else np.nan
            )

            rows.append(
                {
                    "subgroup_type": subgroup_type,
                    "subgroup": subgroup_value,
                    "n": int(len(group_df)),
                    "actual_negative_n": int(len(negatives)),
                    "actual_positive_n": int(len(positives)),
                    "false_positive_rate": round(false_positive_rate, 4)
                    if not np.isnan(false_positive_rate)
                    else np.nan,
                    "false_negative_rate": round(false_negative_rate, 4)
                    if not np.isnan(false_negative_rate)
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def plot_confusion_matrix(frame: pd.DataFrame):
    """Save a confusion matrix figure for the deployed model."""
    y_true = frame["y_true"]
    y_pred = frame["y_pred"]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not depressed", "Depressed"],
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix - Random Forest", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_random_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_error_frame()
    outcome_summary = make_outcome_summary(frame)
    feature_deltas = make_feature_deltas(frame)
    subgroup_rates = make_error_rate_by_subgroup(frame)

    outcome_summary.to_csv(MODELS_DIR / "error_outcome_summary.csv", index=False)
    feature_deltas.to_csv(MODELS_DIR / "error_feature_deltas.csv", index=False)
    subgroup_rates.to_csv(MODELS_DIR / "error_rate_by_subgroup.csv", index=False)
    plot_confusion_matrix(frame)

    print("Saved error-analysis artifacts:")
    print(f"  {MODELS_DIR / 'error_outcome_summary.csv'}")
    print(f"  {MODELS_DIR / 'error_feature_deltas.csv'}")
    print(f"  {MODELS_DIR / 'error_rate_by_subgroup.csv'}")
    print(f"  {FIGURES_DIR / 'confusion_matrix_random_forest.png'}")


if __name__ == "__main__":
    main()
