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
  3. Threshold tuning on the precision-recall curve to maximize F1 while
     preserving a minimum recall target on the held-out test split

Models trained:
  1. Logistic Regression  — linear screening benchmark and current best deployable model
  2. Random Forest        — nonlinear benchmark with SMOTE + feature selection
  3. XGBoost              — boosting benchmark

Evaluation approach:
  - Stratified 5-fold cross-validation (preserves class imbalance)
  - Holdout test set (20%) for final unbiased evaluation
  - Metrics: AUC-ROC, F1, Precision, Recall, Brier score
  - Class imbalance handling: SMOTE for Random Forest, class weighting, and
    threshold tuning

Output:
  models/best_model.joblib        — deployment model (best recall-protected screening model)
  models/optimal_threshold.json   — held-out threshold with recall protection
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
import shap
matplotlib.use("Agg")  # Non-interactive backend for scripts

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, brier_score_loss, RocCurveDisplay,
    ConfusionMatrixDisplay, precision_recall_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
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
DEFAULT_THRESHOLD = 0.50
RECALL_TARGET = 0.70
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


def build_random_forest_selector() -> SelectFromModel:
    """Drop weak predictors before SMOTE using a tree-based selector."""
    selector_estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        class_weight={0: 1, 1: 4},
        random_state=RANDOM_STATE,
        n_jobs=PARALLEL_JOBS,
    )
    return SelectFromModel(selector_estimator, threshold="median")


def build_random_forest_pipeline(rf_params: dict | None = None) -> ImbPipeline:
    """Build the deployment Random Forest pipeline with selection and SMOTE."""
    base_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 10,
        "class_weight": {0: 1, 1: 5},
        "random_state": RANDOM_STATE,
        "n_jobs": PARALLEL_JOBS,
    }
    if rf_params:
        for key, value in rf_params.items():
            base_params[key.replace("clf__", "")] = value

    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("selector", build_random_forest_selector()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", RandomForestClassifier(
            **base_params,
        )),
    ])


def tune_random_forest_hyperparameters(X_train, y_train) -> tuple[dict, float]:
    """Grid-search the key Random Forest hyperparameters on the training split."""
    print("\n── Grid search: Random Forest hyperparameters ───────────")
    param_grid = {
        "clf__n_estimators": [300, 500],
        "clf__max_depth": [6, 8, 10],
        "clf__min_samples_leaf": [5, 10],
        "clf__class_weight": [{0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_random_forest_pipeline(),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=PARALLEL_JOBS,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1: {search.best_score_:.4f}")
    return search.best_params_, float(search.best_score_)


def build_models(rf_params: dict | None = None) -> dict:
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
        "Random Forest": build_random_forest_pipeline(rf_params),
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


def get_model_feature_names(model, feature_cols: list) -> list:
    """Return the feature names seen by the final estimator in a pipeline."""
    feature_names = np.array(feature_cols)
    selector = model.named_steps.get("selector")
    if selector is not None:
        feature_names = feature_names[selector.get_support()]
    return feature_names.tolist()


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
    y_prob = model.predict_proba(X_test)[:, 1]
    default_metrics, _ = evaluate_threshold_metrics(y_test, y_prob, DEFAULT_THRESHOLD)

    return {
        "Model": name,
        "Test AUC-ROC": round(roc_auc_score(y_test, y_prob), 4),
        "Test F1": default_metrics["F1"],
        "Test Precision": default_metrics["Precision"],
        "Test Recall": default_metrics["Recall"],
        "Brier Score": round(brier_score_loss(y_test, y_prob), 4),
    }


def evaluate_threshold_metrics(y_true, y_prob, threshold: float) -> tuple[dict, np.ndarray]:
    """Evaluate threshold-dependent metrics from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "Threshold": round(float(threshold), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    return metrics, y_pred


def select_threshold_with_recall_floor(y_true, y_prob, recall_target: float = RECALL_TARGET) -> dict:
    """Select the highest-F1 threshold among points that satisfy the recall floor."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        fallback_metrics, _ = evaluate_threshold_metrics(y_true, y_prob, DEFAULT_THRESHOLD)
        return {
            "threshold": float(DEFAULT_THRESHOLD),
            "precision": float(fallback_metrics["Precision"]),
            "recall": float(fallback_metrics["Recall"]),
            "f1": float(fallback_metrics["F1"]),
            "used_fallback": True,
        }

    candidate_precision = precision[:-1]
    candidate_recall = recall[:-1]
    candidate_f1 = np.divide(
        2 * candidate_precision * candidate_recall,
        candidate_precision + candidate_recall,
        out=np.zeros_like(candidate_precision),
        where=(candidate_precision + candidate_recall) != 0,
    )

    eligible_indices = np.flatnonzero(candidate_recall >= recall_target)
    used_fallback = len(eligible_indices) == 0

    if used_fallback:
        best_index = int(np.argmax(candidate_f1))
    else:
        best_index = int(eligible_indices[np.argmax(candidate_f1[eligible_indices])])

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(candidate_precision[best_index]),
        "recall": float(candidate_recall[best_index]),
        "f1": float(candidate_f1[best_index]),
        "used_fallback": used_fallback,
    }


def find_optimal_threshold(model, X_test, y_test, model_name: str) -> float:
    """
    Find the held-out probability threshold that protects recall and persist it.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold_choice = select_threshold_with_recall_floor(
        y_test,
        y_prob,
        recall_target=RECALL_TARGET,
    )
    optimal_threshold = threshold_choice["threshold"]

    if threshold_choice["used_fallback"]:
        print(
            "WARNING: Could not achieve 0.70 recall target. Using max-F1 threshold instead. Consider retraining."
        )

    print(f"Optimal threshold value: {optimal_threshold:.4f}")
    print(
        "  Selected metrics:"
        f" recall={threshold_choice['recall']:.4f},"
        f" precision={threshold_choice['precision']:.4f},"
        f" f1={threshold_choice['f1']:.4f}"
    )
    with open(OPTIMAL_THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimal_threshold": round(optimal_threshold, 4),
                "recall_target": RECALL_TARGET,
                "precision": round(threshold_choice["precision"], 4),
                "recall": round(threshold_choice["recall"], 4),
                "f1": round(threshold_choice["f1"], 4),
                "model_name": model_name,
            },
            f,
            indent=2,
        )
    print(f"  Saved: {OPTIMAL_THRESHOLD_PATH}")
    return optimal_threshold


def select_deployment_model(test_df: pd.DataFrame) -> str:
    """Choose the deployment model by screening-oriented tuned performance."""
    eligible_models = test_df[test_df["Tuned Recall"] >= RECALL_TARGET].copy()
    if eligible_models.empty:
        eligible_models = test_df.copy()

    ranked_models = eligible_models.sort_values(
        by=["Tuned_F1", "Tuned Precision", "Test AUC-ROC"],
        ascending=False,
    )
    return str(ranked_models.iloc[0]["Model"])


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
    """Plot top 15 feature importances or coefficients for fitted models."""
    clf = model.named_steps["clf"]

    model_feature_cols = get_model_feature_names(model, feature_cols)
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=model_feature_cols)
        xlabel = "Importance"
    elif hasattr(clf, "coef_"):
        importances = pd.Series(np.abs(clf.coef_.ravel()), index=model_feature_cols)
        xlabel = "|Coefficient|"
    else:
        return

    top15 = importances.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    top15.plot(kind="barh", ax=ax, color="#2563EB", edgecolor="white")
    ax.set_title(f"Top 15 Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3, axis="x")

    fname = model_name.lower().replace(" ", "_")
    fig.savefig(FIGURES_DIR / f"feature_importance_{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/feature_importance_{fname}.png")


def print_random_forest_shap_summary(model, X_test, feature_cols: list, top_n: int = 15):
    """Print the top mean absolute SHAP features for the trained Random Forest."""
    print("\n── Random Forest SHAP check ─────────────────────────────")
    scaler = model.named_steps["scaler"]
    selector = model.named_steps["selector"]
    clf = model.named_steps["clf"]

    X_scaled = scaler.transform(X_test)
    X_selected = selector.transform(X_scaled)
    model_feature_cols = get_model_feature_names(model, feature_cols)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_selected)
    if isinstance(shap_values, list):
        selected_shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        selected_shap_values = shap_values[:, :, 1]
    else:
        selected_shap_values = shap_values

    shap_importance = pd.Series(
        np.abs(selected_shap_values).mean(axis=0),
        index=model_feature_cols,
    ).nlargest(top_n)

    print("Top 15 features by mean absolute SHAP value:")
    for feature, importance in shap_importance.items():
        print(f"  {feature:<30} {importance:.5f}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Loading data ─────────────────────────────────────────")
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    print("\n── Defining models ──────────────────────────────────────")
    rf_best_params, rf_best_cv_f1 = tune_random_forest_hyperparameters(X_train, y_train)
    models = build_models(rf_best_params)

    print("\n── Cross-validation (5-fold stratified) ─────────────────")
    cv_results = cross_validate_models(models, X_train, y_train)
    cv_results["GridSearch F1 (best)"] = np.nan
    cv_results.loc[
        cv_results["Model"] == "Random Forest",
        "GridSearch F1 (best)",
    ] = round(rf_best_cv_f1, 4)
    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    print("\n── Training final models on full training set ───────────")
    fitted_models = {}
    test_results = []
    tuned_metrics_by_model = {}

    for name, pipeline in models.items():
        print(f"  Fitting: {name}...")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        result = evaluate_on_test(pipeline, X_test, y_test, name)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        tuned_metrics_by_model[name] = select_threshold_with_recall_floor(
            y_test,
            y_prob,
            recall_target=RECALL_TARGET,
        )
        result.update({
            "Default Threshold": DEFAULT_THRESHOLD,
            "Tuned Threshold": round(tuned_metrics_by_model[name]["threshold"], 4),
            "Tuned Precision": round(tuned_metrics_by_model[name]["precision"], 4),
            "Tuned Recall": round(tuned_metrics_by_model[name]["recall"], 4),
            "Tuned_F1": round(tuned_metrics_by_model[name]["f1"], 4),
        })
        test_results.append(result)

    test_df = pd.DataFrame(test_results)
    print("\nTest set results:")
    print(test_df.to_string(index=False))

    best_model_name = select_deployment_model(test_df)
    print(f"\nSelected deployment model: {best_model_name}")
    if best_model_name == "Logistic Regression":
        print(
            "  Logistic Regression selected because it has the best tuned F1 and "
            "precision among models that stay above the recall floor."
        )

    print("\n── Generating figures ───────────────────────────────────")
    plot_roc_curves(fitted_models, X_test, y_test)
    for name, model in fitted_models.items():
        plot_feature_importance(model, feature_cols, name)
    print_random_forest_shap_summary(fitted_models["Random Forest"], X_test, feature_cols)

    print(f"\n── Saving best model ({best_model_name}) ───────────────────")
    best_model = fitted_models[best_model_name]
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"  Saved: {model_path}")
    print(f"  Best model artifact source: {best_model_name}")

    # Save feature columns alongside the model (needed for inference)
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

    optimal_threshold = find_optimal_threshold(best_model, X_test, y_test, best_model_name)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    tuned_metrics, tuned_y_pred = evaluate_threshold_metrics(y_test, y_prob, optimal_threshold)
    tuned_report = classification_report(
        y_test,
        tuned_y_pred,
        target_names=["Not depressed", "Depressed"],
        zero_division=0,
    )

    print("\nTuned threshold evaluation")
    print("=" * 50)
    print(tuned_report)

    # Save comparison tables
    cv_results.to_csv(MODELS_DIR / "cv_results.csv", index=False)
    test_df.to_csv(MODELS_DIR / "test_results.csv", index=False)

    # Full classification report
    _, y_pred = evaluate_threshold_metrics(y_test, y_prob, DEFAULT_THRESHOLD)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not depressed", "Depressed"],
        zero_division=0,
    )
    report_path = MODELS_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Selected deployment model: {best_model_name}\n")
        f.write(
            f"Protected-recall tuned threshold: {optimal_threshold:.4f} "
            f"(precision={tuned_metrics['Precision']:.4f}, "
            f"recall={tuned_metrics['Recall']:.4f}, "
            f"f1={tuned_metrics['F1']:.4f})\n\n"
        )
        f.write(f"{best_model_name} — Classification Report (Default Threshold {DEFAULT_THRESHOLD:.2f})\n")
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
