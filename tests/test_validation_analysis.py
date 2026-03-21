import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validation_analysis import (
    bootstrap_metric_intervals,
    make_subgroup_metrics,
)


def test_bootstrap_metric_intervals_returns_ordered_bounds():
    group_df = pd.DataFrame(
        {
            "y_true": [0, 1] * 20,
            "y_prob": [0.1, 0.8, 0.2, 0.7] * 10,
        }
    )

    result = bootstrap_metric_intervals(
        group_df,
        n_bootstrap=40,
        random_state=123,
    )

    assert result["bootstrap_replicates"] == 40
    for metric in ["auc_roc", "precision", "recall", "f1"]:
        assert 0.0 <= result[f"{metric}_ci_low"] <= result[f"{metric}_ci_high"] <= 1.0


def test_make_subgroup_metrics_includes_confidence_interval_columns():
    groups = pd.DataFrame(
        {
            "SEQN": list(range(1, 21)),
            "age": [25] * 10 + [55] * 10,
            "sex_female": [0, 1] * 10,
            "poverty_ratio": [0.8] * 10 + [2.5] * 10,
            "race_eth": [3, 4] * 10,
        }
    )
    y_true = pd.Series([0, 1, 0, 1, 0] * 4)
    y_prob = np.array([0.2, 0.7, 0.3, 0.8, 0.4] * 4)

    subgroup_metrics = make_subgroup_metrics(
        groups,
        y_true,
        y_prob,
        n_bootstrap=25,
        random_state=99,
    )

    expected_columns = {
        "auc_roc_ci_low",
        "auc_roc_ci_high",
        "precision_ci_low",
        "precision_ci_high",
        "recall_ci_low",
        "recall_ci_high",
        "f1_ci_low",
        "f1_ci_high",
        "bootstrap_replicates",
    }

    assert expected_columns.issubset(subgroup_metrics.columns)
    assert (subgroup_metrics["bootstrap_replicates"] == 25).all()
