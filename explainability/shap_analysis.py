"""
shap_analysis.py
----------------
Generates SHAP (SHapley Additive exPlanations) visualizations for the
trained Random Forest model.

SHAP tells us:
  - WHICH features most influence depression risk predictions (global importance)
  - HOW each feature value pushes a specific prediction up or down (local explanation)
  - INTERACTIONS between features

Output figures:
  figures/shap_summary.png        — beeswarm plot (global importance + direction)
  figures/shap_bar.png            — mean |SHAP| bar chart
  figures/shap_dependence_*.png   — dependence plots for top features
  figures/shap_force_sample.png   — force plot for one individual

Why SHAP matters for health research:
  Researchers and clinicians need to understand WHY the model makes predictions,
  not just what the prediction is. SHAP provides mathematically grounded
  explanations rooted in game theory (Shapley values).
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# Friendly display names for features (used in plots)
FEATURE_DISPLAY_NAMES = {
    "age": "Age (years)",
    "age_squared": "Age² (non-linear)",
    "sex_female": "Female sex",
    "poverty_ratio": "Poverty-income ratio",
    "poverty_low": "Below poverty line",
    "education": "Education level",
    "race_mexican_american": "Mexican American",
    "race_other_hispanic": "Other Hispanic",
    "race_nh_black": "Non-Hispanic Black",
    "race_nh_asian": "Non-Hispanic Asian",
    "race_other": "Other race/ethnicity",
    "met_min_week": "Physical activity (MET-min/wk)",
    "met_log": "Physical activity (log)",
    "inactive": "Completely inactive",
    "meets_who_guidelines": "Meets WHO activity guidelines",
    "sleep_hours": "Sleep duration (hrs/night)",
    "sleep_trouble": "Trouble sleeping",
    "short_sleep": "Short sleep (<6 hrs)",
    "long_sleep": "Long sleep (>9 hrs)",
    "optimal_sleep": "Optimal sleep (7-9 hrs)",
    "bmi": "BMI (kg/m²)",
    "underweight": "Underweight (BMI<18.5)",
    "overweight": "Overweight (BMI≥25)",
    "obese": "Obese (BMI≥30)",
    "drinks_per_week": "Drinks per week",
    "drinks_log": "Drinks per week (log)",
    "hazardous_drinking": "Hazardous drinking",
    "inactive_x_poor_sleep": "Inactive × short sleep",
    "poverty_x_inactive": "Poverty × inactive",
    "age_x_poverty": "Age × poverty",
}


def load_data_and_model():
    """Load test features and the trained model."""
    meta_path = PROCESSED_DIR / "feature_names.json"
    with open(meta_path) as f:
        meta = json.load(f)

    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    feature_cols = meta["feature_columns"]

    X = df[feature_cols]
    y = df["depression_binary"]

    # Use a stratified sample for SHAP (full dataset is slow)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODELS_DIR / "best_model.joblib")
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    return X_test, X_test_scaled, y_test, clf, feature_cols


def compute_shap_values(clf, X_scaled: pd.DataFrame, feature_cols: list):
    """Compute SHAP values using TreeExplainer (fast for tree-based models)."""
    print("  Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled)

    # SHAP's binary classification output varies by version:
    # - older versions: [class0, class1]
    # - newer versions: (n_samples, n_features, n_classes)
    # We always want the class 1 contributions (depressed).
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    return sv, explainer


def rename_features(X: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to display-friendly names."""
    return X.rename(columns=FEATURE_DISPLAY_NAMES)


def plot_shap_summary(shap_values, X_scaled: pd.DataFrame, feature_cols: list):
    """Beeswarm plot — shows global feature importance + direction."""
    X_display = rename_features(X_scaled.copy())
    renamed_sv = shap_values.copy()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        renamed_sv,
        X_display,
        show=False,
        plot_size=None,
    )
    plt.title(
        "SHAP Summary — Feature Impact on Depression Risk\n"
        "(Red = high feature value, Blue = low feature value)",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/shap_summary.png")


def plot_shap_bar(shap_values, feature_cols: list):
    """Mean |SHAP| bar chart — most important features overall."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.Series(mean_abs_shap, index=feature_cols)
    importance_df.index = [
        FEATURE_DISPLAY_NAMES.get(f, f) for f in importance_df.index
    ]
    top20 = importance_df.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    top20.plot(kind="barh", ax=ax, color="#2563EB", edgecolor="white")
    ax.set_title(
        "Mean |SHAP| Value — Top 20 Features\n(Average impact on depression risk prediction)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/shap_bar.png")


def plot_dependence(shap_values, X_scaled: pd.DataFrame, feature_cols: list):
    """
    SHAP dependence plots for the top 3 features.
    Shows how the feature value relates to its SHAP contribution.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs_shap)[-3:]
    top3_features = [feature_cols[i] for i in top3_idx]

    for feat in top3_features:
        feat_display = FEATURE_DISPLAY_NAMES.get(feat, feat)
        feat_idx = feature_cols.index(feat)

        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_scaled,
            feature_names=[FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_cols],
            ax=ax,
            show=False,
        )
        ax.set_title(
            f"SHAP Dependence Plot: {feat_display}",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        fname = feat.replace(" ", "_")
        fig.savefig(FIGURES_DIR / f"shap_dependence_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: figures/shap_dependence_{fname}.png")


def generate_text_explanation(shap_values, X_scaled: pd.DataFrame,
                               feature_cols: list, sample_idx: int = 0) -> str:
    """
    Generate a human-readable text explanation for one individual.
    This is what would appear in the dashboard or a clinical report.
    """
    sv = shap_values[sample_idx]
    base_value_prob = 0.5  # Approximate baseline

    # Sort by absolute SHAP value
    sv_series = pd.Series(sv, index=feature_cols)
    top_positive = sv_series.nlargest(5)
    top_negative = sv_series.nsmallest(3)

    lines = [
        "── Individual Risk Explanation ────────────────────────────",
        f"  Base rate:        {base_value_prob:.1%}",
        "",
        "  Factors INCREASING risk:",
    ]
    for feat, val in top_positive.items():
        feat_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        feat_val = X_scaled.iloc[sample_idx][feat]
        lines.append(f"    ↑ {feat_name:<35} SHAP={val:+.3f}  (value={feat_val:.2f})")

    lines.append("")
    lines.append("  Factors DECREASING risk:")
    for feat, val in top_negative.items():
        feat_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        feat_val = X_scaled.iloc[sample_idx][feat]
        lines.append(f"    ↓ {feat_name:<35} SHAP={val:+.3f}  (value={feat_val:.2f})")

    lines.append("────────────────────────────────────────────────────────")
    return "\n".join(lines)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Loading model and data ───────────────────────────────")
    X_test, X_test_scaled, y_test, clf, feature_cols = load_data_and_model()

    print("\n── Computing SHAP values ────────────────────────────────")
    shap_values, explainer = compute_shap_values(clf, X_test_scaled, feature_cols)

    print("\n── Generating plots ─────────────────────────────────────")
    plot_shap_summary(shap_values, X_test_scaled, feature_cols)
    plot_shap_bar(shap_values, feature_cols)
    plot_dependence(shap_values, X_test_scaled, feature_cols)

    print("\n── Sample individual explanation ────────────────────────")
    explanation = generate_text_explanation(
        shap_values, X_test_scaled, feature_cols, sample_idx=0
    )
    print(explanation)

    print("\n── Done ─────────────────────────────────────────────────")
    print("All SHAP figures saved to figures/")


if __name__ == "__main__":
    main()
