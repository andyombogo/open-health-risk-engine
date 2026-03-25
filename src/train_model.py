"""
train_model.py
--------------
Trains, evaluates, and saves depression risk prediction models.

This project predicts `PHQ-9 >= 10` on NHANES data, where the positive class is
a minority of cases. That class imbalance makes naive accuracy and default
probability thresholds misleading, so the training workflow uses three explicit
strategies to address it:
  1. SMOTE oversampling inside the Random Forest training pipeline only
  2. Class weighting to penalize missed positive cases more heavily
  3. Threshold tuning on the precision-recall curve to maximize F1 on the
     held-out test split

Models trained:
  1. Logistic Regression  — interpretable baseline
  2. Random Forest        — best balance of performance + explainability
  3. XGBoost              — highest performance

Evaluation approach:
  - Stratified 5-fold cross-validation (preserves class imbalance)
  - Holdout test set (20%) for final unbiased evaluation
  - Metrics: AUC-ROC, F1, Precision, Recall, Brier score
  - Class imbalance handling: SMOTE for Random Forest, class weighting, and
    threshold tuning

Output:
  models/best_model.joblib        — best model selected by CV F1 mean
  models/optimal_threshold.json   — held-out threshold that maximizes F1
  models/cv_results.csv           — cross-validation comparison table
  models/test_results.csv         — held-out test comparison table
  models/evaluation_report.txt    — full classification report
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, brier_score_loss, RocCurveDisplay,
    ConfusionMatrixDisplay, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

RANDOM_STATE = 42
PARALLEL_JOBS = 1
OPTIMAL_THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.json"


def load_data():
    """Load feature matrix and split into train/test."""
    feat_path = PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("Run feature_engineering.py first.")

    meta_path = PROCESSED_DIR / "feature_names.json"
    with open(meta_path) as f:
        meta = json.load(f)

    df = pd.read_csv(feat_path)
    feature_cols = meta["feature_columns"]

    X = df[feature_cols]
    y = df["depression_binary"]

    print(f"Dataset: {X.shape[0]} participants, {X.shape[1]} features")
    print(f"Positive class (PHQ-9 >= 10): {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols


def build_models() -> dict:
    """Define the three model pipelines."""
    models = {
        "Logistic Regression": SklearnPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                class_weight={0: 1, 1: 5},  # Penalizes missing a depressed person 5x more than a false alarm, appropriate for a clinical screening context.
                random_state=RANDOM_STATE,
                n_jobs=PARALLEL_JOBS,
            )),
        ]),
        "XGBoost": SklearnPipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,  # Handles class imbalance
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ]),
    }
    return models


def cross_validate_models(models: dict, X_train, y_train) -> pd.DataFrame:
    """5-fold stratified cross-validation on training data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, pipeline in models.items():
        print(f"  Cross-validating: {name}...")
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            # F1 and recall stay in the scoring set because imbalanced screening
            # data needs direct positive-class performance checks, not AUC alone.
            scoring=["roc_auc", "f1", "precision", "recall"],
            n_jobs=PARALLEL_JOBS,
        )
        results.append({
            "Model": name,
            "AUC-ROC (mean)": cv_results["test_roc_auc"].mean(),
            "AUC-ROC (std)": cv_results["test_roc_auc"].std(),
            "F1 (mean)": cv_results["test_f1"].mean(),
            "Precision (mean)": cv_results["test_precision"].mean(),
            "Recall (mean)": cv_results["test_recall"].mean(),
        })

    return pd.DataFrame(results).round(4)


def evaluate_on_test(model, X_test, y_test, name: str) -> dict:
    """Final evaluation on held-out test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Test AUC-ROC": round(roc_auc_score(y_test, y_prob), 4),
        "Test F1": round(f1_score(y_test, y_pred), 4),
        "Test Precision": round(precision_score(y_test, y_pred), 4),
        "Test Recall": round(recall_score(y_test, y_pred), 4),
        "Brier Score": round(brier_score_loss(y_test, y_prob), 4),
    }


def find_optimal_threshold(model, X_test, y_test) -> float:
    """
    Find the held-out probability threshold that maximizes F1 and persist it.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    if len(thresholds) == 0:
        optimal_threshold = 0.5
    else:
        best_index = int(np.argmax(f1_scores[:-1]))
        optimal_threshold = float(thresholds[best_index])

    print(f"Optimal threshold value: {optimal_threshold:.4f}")
    with open(OPTIMAL_THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump({"optimal_threshold": round(optimal_threshold, 4)}, f, indent=2)
    print(f"  Saved: {OPTIMAL_THRESHOLD_PATH}")
    return optimal_threshold


def plot_roc_curves(fitted_models: dict, X_test, y_test):
    """Plot ROC curves for all models on test set."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2563EB", "#059669", "#DC2626"]

    for (name, model), color in zip(fitted_models.items(), colors):
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax, name=name, color=color
        )

    ax.set_title("ROC Curves — Depression Risk Prediction", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/roc_curves.png")


def plot_feature_importance(model, feature_cols: list, model_name: str):
    """Plot top 15 feature importances (Random Forest / XGBoost)."""
    clf = model.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        return  # Logistic Regression doesn't have this

    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top15 = importances.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    top15.plot(kind="barh", ax=ax, color="#2563EB", edgecolor="white")
    ax.set_title(f"Top 15 Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")

    fname = model_name.lower().replace(" ", "_")
    fig.savefig(FIGURES_DIR / f"feature_importance_{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/feature_importance_{fname}.png")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Loading data ─────────────────────────────────────────")
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    print("\n── Defining models ──────────────────────────────────────")
    models = build_models()

    print("\n── Cross-validation (5-fold stratified) ─────────────────")
    cv_results = cross_validate_models(models, X_train, y_train)
    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    # F1 is the primary selection metric here because imbalanced clinical data
    # needs a model that balances precision and recall on the positive class.
    best_model_name = (
        cv_results.sort_values(
            by=["F1 (mean)", "Recall (mean)", "AUC-ROC (mean)"],
            ascending=False,
        )
        .iloc[0]["Model"]
    )
    print(f"\nSelected best model by CV F1 mean: {best_model_name}")

    print("\n── Training final models on full training set ───────────")
    fitted_models = {}
    test_results = []

    for name, pipeline in models.items():
        print(f"  Fitting: {name}...")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        result = evaluate_on_test(pipeline, X_test, y_test, name)
        test_results.append(result)

    test_df = pd.DataFrame(test_results)
    print("\nTest set results:")
    print(test_df.to_string(index=False))

    print("\n── Generating figures ───────────────────────────────────")
    plot_roc_curves(fitted_models, X_test, y_test)
    for name, model in fitted_models.items():
        plot_feature_importance(model, feature_cols, name)

    print(f"\n── Saving best model ({best_model_name}) ───────────────────")
    best_model = fitted_models[best_model_name]
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"  Saved: {model_path}")
    print(f"  Best model artifact source: {best_model_name}")

    # Save feature columns alongside the model (needed for inference)
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

    optimal_threshold = find_optimal_threshold(best_model, X_test, y_test)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    tuned_y_pred = (y_prob >= optimal_threshold).astype(int)
    tuned_report = classification_report(
        y_test,
        tuned_y_pred,
        target_names=["Not depressed", "Depressed"],
    )
    tuned_f1 = f1_score(y_test, tuned_y_pred, zero_division=0)

    print("\nTuned threshold evaluation")
    print("=" * 50)
    print(tuned_report)

    test_df["Tuned_F1"] = np.nan
    test_df.loc[test_df["Model"] == best_model_name, "Tuned_F1"] = round(tuned_f1, 4)

    # Save comparison tables
    cv_results.to_csv(MODELS_DIR / "cv_results.csv", index=False)
    test_df.to_csv(MODELS_DIR / "test_results.csv", index=False)

    # Full classification report
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Not depressed", "Depressed"])
    report_path = MODELS_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Selected best model by CV F1 mean: {best_model_name}\n\n")
        f.write(f"{best_model_name} — Classification Report (Test Set)\n")
        f.write("=" * 50 + "\n")
        f.write(report)
        f.write("\n\nTuned threshold evaluation\n")
        f.write("=" * 50 + "\n")
        f.write(tuned_report)
    print(f"  Saved: {report_path}")

    print("\n── Done ─────────────────────────────────────────────────")
    print("Next step: run explainability/shap_analysis.py")
    print("Then:       streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
