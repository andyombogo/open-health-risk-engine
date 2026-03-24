"""Baseline NLP pipeline for Phase 4 clinical-note experiments.

This track is intentionally isolated from the deployed NHANES calculator.
It trains a lightweight TF-IDF plus Logistic Regression baseline on any
CSV that contains note text and a binary label column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.nlp.preprocessing import prepare_note_dataframe

RANDOM_STATE = 42
TEXT_FEATURE_COLUMN = "note_text"
NUMERIC_FEATURE_COLUMNS = [
    "section_count",
    "word_count",
    "has_history_section",
    "has_assessment_section",
    "has_plan_section",
    "antidepressant_mention_count",
    "fatigue_mention_count",
    "hopelessness_mention_count",
    "sleep_issue_mention_count",
    "symptom_mention_count",
    "has_antidepressant_mention",
    "has_fatigue_mention",
    "has_hopelessness_mention",
    "has_sleep_issue_mention",
]


def build_baseline_pipeline(max_features: int = 5000) -> Pipeline:
    """Build a lightweight text-plus-structured baseline model."""
    features = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=max_features,
                ),
                TEXT_FEATURE_COLUMN,
            ),
            ("numeric", MaxAbsScaler(), NUMERIC_FEATURE_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _evaluate_binary_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute standard holdout metrics for the baseline text classifier."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def train_baseline_text_model(
    frame: pd.DataFrame,
    text_column: str = "note_text",
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Train and evaluate a baseline text model from a note-level DataFrame."""
    prepared = prepare_note_dataframe(
        frame,
        text_column=text_column,
        label_column=label_column,
    )

    if prepared[label_column].nunique() < 2:
        raise ValueError("The label column must contain at least two classes.")
    if len(prepared) < 6:
        raise ValueError("Provide at least 6 note rows for a usable split.")

    X = prepared[[text_column, *NUMERIC_FEATURE_COLUMNS]].copy()
    y = prepared[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_baseline_pipeline()
    model.fit(X_train, y_train)
    metrics = _evaluate_binary_classifier(model, X_test, y_test)

    return {
        "model": model,
        "metrics": {key: round(value, 4) for key, value in metrics.items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_rows": int(len(prepared)),
        "text_column": text_column,
        "label_column": label_column,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Phase 4 baseline note-text classifier from CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Path to the note CSV.")
    parser.add_argument(
        "--text-column",
        default="note_text",
        help="Name of the free-text note column.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the binary label column.",
    )
    parser.add_argument(
        "--output-model",
        default="models/nlp_baseline.joblib",
        help="Where to save the trained baseline model.",
    )
    parser.add_argument(
        "--output-metrics",
        default="models/nlp_baseline_metrics.json",
        help="Where to save the evaluation summary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_model_path = Path(args.output_model)
    output_metrics_path = Path(args.output_metrics)

    frame = pd.read_csv(input_path)
    results = train_baseline_text_model(
        frame,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["model"], output_model_path)
    metrics_payload = {
        key: value for key, value in results.items() if key != "model"
    }
    output_metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Saved model to {output_model_path}")
    print(f"Saved metrics to {output_metrics_path}")
    print(json.dumps(metrics_payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
