"""
validation_analysis.py
----------------------
Generate calibration, threshold, precision-recall, and subgroup evaluation
artifacts for the deployed model using the same held-out split definition as
train_model.py.

Outputs:
  models/calibration_table.csv
  models/threshold_metrics.csv
  models/subgroup_metrics.csv (with bootstrap confidence intervals)
  figures/calibration_curve_random_forest.png
  figures/precision_recall_curve_random_forest.png
  figures/threshold_tradeoffs_random_forest.png
"""

from pathlib import Path
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict_risk import (  # noqa: E402
    DEFAULT_DECISION_THRESHOLD,
    resolve_decision_threshold,
    resolve_model_path,
)

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BOOTSTRAP_REPLICATES = 300

RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / Multiracial",
}


def describe_model_artifact(model_path: Path) -> str:
    """Return a human-readable label for the currently evaluated artifact."""
    name = model_path.name.lower()
    if name == "best_model.joblib":
        threshold_path = MODELS_DIR / "optimal_threshold.json"
        if threshold_path.exists():
            try:
                with open(threshold_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                model_name = str(payload.get("model_name", "")).strip()
                if model_name:
                    return model_name
            except (OSError, ValueError, TypeError):
                pass
    if "logistic" in name:
        return "Logistic Regression"
    if "xgboost" in name or "xgb" in name:
        return "XGBoost"
    if "random_forest" in name or "forest" in name:
        if "calibrated" in name:
            return "Sigmoid-calibrated Random Forest"
        return "Random Forest"
    if "best_model" in name:
        return "Current deployment model"
    return model_path.stem.replace("_", " ").title()


def safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Return ROC AUC if both classes are present, otherwise NaN."""
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_binary_metrics(
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict:
    """Compute core classification metrics from probabilities."""
    y_true = pd.Series(y_true).astype(int)
    y_prob = pd.Series(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "positive_rate": float(y_true.mean()),
        "mean_predicted_probability": float(y_prob.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
        "auc_roc": safe_auc(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def percentile_interval(values: list[float], alpha: float = 0.05) -> tuple:
    """Return a percentile interval for a numeric bootstrap sample."""
    valid_values = np.asarray([value for value in values if not np.isnan(value)])
    if len(valid_values) == 0:
        return np.nan, np.nan

    lower = float(np.quantile(valid_values, alpha / 2))
    upper = float(np.quantile(valid_values, 1 - alpha / 2))
    return lower, upper


def bootstrap_metric_intervals(
    group_df: pd.DataFrame,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Estimate subgroup metric intervals with bootstrap resampling."""
    tracked_metrics = ["auc_roc", "precision", "recall", "f1"]
    bootstrap_values = {metric: [] for metric in tracked_metrics}
    rng = np.random.default_rng(random_state)

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(0, len(group_df), len(group_df))
        sampled = group_df.iloc[sampled_indices]
        sampled_metrics = compute_binary_metrics(
            sampled["y_true"],
            sampled["y_prob"],
            threshold=threshold,
        )
        for metric in tracked_metrics:
            bootstrap_values[metric].append(sampled_metrics[metric])

    intervals = {"bootstrap_replicates": int(n_bootstrap)}
    for metric in tracked_metrics:
        lower, upper = percentile_interval(bootstrap_values[metric])
        intervals[f"{metric}_ci_low"] = round(lower, 4) if not np.isnan(lower) else np.nan
        intervals[f"{metric}_ci_high"] = round(upper, 4) if not np.isnan(upper) else np.nan

    return intervals


def load_validation_data(model_path: Path | None = None):
    """Load the feature matrix, subgroup fields, and fitted pipeline."""
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    nhanes = pd.read_csv(PROCESSED_DIR / "nhanes_clean.csv", usecols=["SEQN", "race_eth"])
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    resolved_model_path = resolve_model_path(model_path)
    model = joblib.load(resolved_model_path)

    frame = features.merge(nhanes, on="SEQN", how="left")
    X = frame[feature_cols]
    y = frame["depression_binary"].astype(int)
    groups = frame[["SEQN", "age", "sex_female", "poverty_ratio", "race_eth"]].copy()
    return model, X, y, groups


def make_calibration_table(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Create an equal-width calibration table across 10 bins."""
    calibration = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    calibration["bin"] = pd.cut(
        calibration["y_prob"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        duplicates="drop",
    )

    table = (
        calibration.groupby("bin", observed=False)
        .agg(
            sample_count=("y_true", "size"),
            mean_predicted_probability=("y_prob", "mean"),
            observed_positive_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    table = table[table["sample_count"] > 0].copy()
    table["calibration_gap"] = (
        table["observed_positive_rate"] - table["mean_predicted_probability"]
    )
    table["bin"] = table["bin"].astype(str)
    return table.round(4)


def make_threshold_table(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Evaluate common operating thresholds for precision, recall, and F1."""
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(
                    precision_score(y_true, y_pred, zero_division=0), 4
                ),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "predicted_positive_rate": round(float(y_pred.mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def make_subgroup_metrics(
    groups: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    model_artifact: str | None = None,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute subgroup metrics for sex, age band, poverty band, and race."""
    frame = groups.copy()
    frame["y_true"] = y_true.to_numpy()
    frame["y_prob"] = y_prob
    frame["y_pred"] = (frame["y_prob"] >= threshold).astype(int)

    frame["sex_group"] = np.where(frame["sex_female"] == 1, "Female", "Male")
    frame["age_band"] = pd.cut(
        frame["age"],
        bins=[18, 35, 50, 65, np.inf],
        right=False,
        labels=["18-34", "35-49", "50-64", "65+"],
    ).astype(str)
    frame["poverty_band"] = pd.cut(
        frame["poverty_ratio"],
        bins=[-np.inf, 1.0, 2.0, np.inf],
        right=False,
        labels=["<1.0", "1.0-1.99", "2.0+"],
    ).astype(str)
    frame["race_group"] = frame["race_eth"].map(RACE_LABELS).fillna("Unknown")

    rows = []
    subgroup_specs = [
        ("sex", "sex_group"),
        ("age_band", "age_band"),
        ("poverty_band", "poverty_band"),
        ("race", "race_group"),
    ]

    for subgroup_type, column in subgroup_specs:
        for subgroup_value, group_df in frame.groupby(column, observed=False):
            point_metrics = compute_binary_metrics(
                group_df["y_true"],
                group_df["y_prob"],
                threshold=threshold,
            )
            bootstrap_intervals = bootstrap_metric_intervals(
                group_df,
                threshold=threshold,
                n_bootstrap=n_bootstrap,
                random_state=random_state + len(rows),
            )
            rows.append(
                {
                    "model_artifact": model_artifact or "",
                    "operating_threshold": round(float(threshold), 4),
                    "subgroup_type": subgroup_type,
                    "subgroup": subgroup_value,
                    "n": int(len(group_df)),
                    "positive_rate": round(point_metrics["positive_rate"], 4),
                    "mean_predicted_probability": round(
                        point_metrics["mean_predicted_probability"], 4
                    ),
                    "predicted_positive_rate": round(
                        point_metrics["predicted_positive_rate"], 4
                    ),
                    "auc_roc": round(point_metrics["auc_roc"], 4)
                    if not np.isnan(point_metrics["auc_roc"])
                    else np.nan,
                    "precision": round(point_metrics["precision"], 4),
                    "recall": round(point_metrics["recall"], 4),
                    "f1": round(point_metrics["f1"], 4),
                    **bootstrap_intervals,
                }
            )

    return pd.DataFrame(rows)


def plot_calibration(table: pd.DataFrame, model_label: str):
    """Save a calibration curve figure."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#94a3b8",
        label="Perfect calibration",
    )
    ax.plot(
        table["mean_predicted_probability"],
        table["observed_positive_rate"],
        marker="o",
        color="#2563eb",
        linewidth=2,
        label=model_label,
    )
    ax.set_title(
        f"Calibration Curve - {model_label}",
        fontweight="bold",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(
        FIGURES_DIR / "calibration_curve_random_forest.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_precision_recall(y_true: pd.Series, y_prob: np.ndarray, model_label: str):
    """Save a precision-recall curve figure."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, color="#059669", linewidth=2)
    ax.set_title(
        (
            "Precision-Recall Curve - "
            f"{model_label} (AP={average_precision:.3f})"
        ),
        fontweight="bold",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.25)
    fig.savefig(
        FIGURES_DIR / "precision_recall_curve_random_forest.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_threshold_tradeoffs(threshold_table: pd.DataFrame, model_label: str):
    """Save threshold-vs-metric tradeoffs for precision, recall, and F1."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        threshold_table["threshold"],
        threshold_table["precision"],
        marker="o",
        color="#2563eb",
        label="Precision",
    )
    ax.plot(
        threshold_table["threshold"],
        threshold_table["recall"],
        marker="o",
        color="#dc2626",
        label="Recall",
    )
    ax.plot(
        threshold_table["threshold"],
        threshold_table["f1"],
        marker="o",
        color="#d97706",
        label="F1",
    )
    ax.set_title(f"Threshold Tradeoffs - {model_label}", fontweight="bold")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(
        FIGURES_DIR / "threshold_tradeoffs_random_forest.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    operating_threshold = resolve_decision_threshold()
    model_path = resolve_model_path()
    model_label = describe_model_artifact(model_path)

    model, X, y, groups = load_validation_data(model_path)
    _, X_test, _, y_test, _, groups_test = train_test_split(
        X,
        y,
        groups,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= operating_threshold).astype(int)

    calibration_table = make_calibration_table(y_test, y_prob)
    threshold_table = make_threshold_table(y_test, y_prob)
    subgroup_metrics = make_subgroup_metrics(
        groups_test,
        y_test,
        y_prob,
        threshold=operating_threshold,
        model_artifact=model_path.name,
    )

    calibration_table.to_csv(MODELS_DIR / "calibration_table.csv", index=False)
    threshold_table.to_csv(MODELS_DIR / "threshold_metrics.csv", index=False)
    subgroup_metrics.to_csv(MODELS_DIR / "subgroup_metrics.csv", index=False)

    plot_calibration(calibration_table, model_label)
    plot_precision_recall(y_test, y_prob, model_label)
    plot_threshold_tradeoffs(threshold_table, model_label)

    print("Saved validation artifacts:")
    print(f"  {MODELS_DIR / 'calibration_table.csv'}")
    print(f"  {MODELS_DIR / 'threshold_metrics.csv'}")
    print(f"  {MODELS_DIR / 'subgroup_metrics.csv'}")
    print(f"  {FIGURES_DIR / 'calibration_curve_random_forest.png'}")
    print(f"  {FIGURES_DIR / 'precision_recall_curve_random_forest.png'}")
    print(f"  {FIGURES_DIR / 'threshold_tradeoffs_random_forest.png'}")
    print(f"\nModel artifact: {model_path.name}")
    print(f"Operating threshold: {operating_threshold:.2f}")
    print("\nOverall metrics on held-out test split:")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Average precision: {average_precision_score(y_test, y_prob):.4f}")
    print(f"  Brier score: {brier_score_loss(y_test, y_prob):.4f}")
    print(
        f"  Precision@{operating_threshold:.2f}: "
        f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
    )
    print(
        f"  Recall@{operating_threshold:.2f}: "
        f"{recall_score(y_test, y_pred, zero_division=0):.4f}"
    )
    print(
        f"  F1@{operating_threshold:.2f}: "
        f"{f1_score(y_test, y_pred, zero_division=0):.4f}"
    )


if __name__ == "__main__":
    main()
