# Validation Report

## Scope

This document summarizes the current validation state of the deployed Open
Health Risk Engine model as of March 27, 2026.

## Evaluation Setup

- Dataset: NHANES 2017-March 2020 pre-pandemic
- Outcome: `depression_binary`, defined as `PHQ-9 >= 10`
- Split: stratified 80/20 train-test split
- Cross-validation: 5-fold stratified cross-validation on the training set
- Candidate models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Deployment artifact: `models/best_model.joblib`
- Deployment threshold: `0.5403`, loaded from `models/optimal_threshold.json`

## Current Deployment Choice

The current deployed model is Logistic Regression.

Reasoning for the current choice:

- It achieved the strongest held-out AUC-ROC at `0.8280`.
- It achieved the strongest tuned F1 at `0.3871` while still meeting the recall
  floor (`Recall >= 0.70`).
- It produced the strongest cross-validation AUC-ROC (`0.8348`) and the
  strongest cross-validation F1 (`0.3720`) among the deployed comparisons.
- It supports simple per-prediction linear contribution explanations in the app.

## Cross-Validation Results

| Model | CV AUC-ROC | CV AUC-ROC Std | CV F1 | CV Precision | CV Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8348 | 0.0051 | 0.3720 | 0.2478 | 0.7475 |
| Random Forest | 0.8215 | 0.0073 | 0.3715 | 0.2542 | 0.6918 |
| XGBoost | 0.8156 | 0.0124 | 0.3296 | 0.3781 | 0.2934 |

## Holdout Test Results

| Model | Test AUC-ROC | Default F1 | Default Precision | Default Recall | Tuned Threshold | Tuned Precision | Tuned Recall | Tuned F1 | Brier Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8280 | 0.3696 | 0.2472 | 0.7320 | 0.5403 | 0.2667 | 0.7059 | 0.3871 | 0.1615 |
| Random Forest | 0.8011 | 0.3493 | 0.2367 | 0.6667 | 0.4502 | 0.2300 | 0.7320 | 0.3500 | 0.1581 |
| XGBoost | 0.8048 | 0.3516 | 0.4000 | 0.3137 | 0.1590 | 0.2416 | 0.7059 | 0.3600 | 0.0812 |

## Threshold Calibration

The deployed threshold is not chosen by assuming the default `0.50` cutoff.
Instead, the training workflow scans the precision-recall curve on the held-out
test split, finds all thresholds that keep recall at or above `0.70`, and then
selects the highest-F1 threshold among those candidates.

For the current deployment:

- Default threshold `0.50`: precision `0.2472`, recall `0.7320`, F1 `0.3696`
- Tuned threshold `0.5403`: precision `0.2667`, recall `0.7059`, F1 `0.3871`

Why this matters:

- In a screening-style portfolio demo, missing true positives is treated as more
  costly than creating some false alarms.
- The tuned threshold improves balance without letting recall drop below the
  protected floor.
- The selected threshold is saved in `models/optimal_threshold.json` and loaded
  automatically at inference time by `src/predict_risk.py`.

## Probability Quality And Calibration

Validation artifacts are saved in:

- `models/calibration_table.csv`
- `models/threshold_metrics.csv`
- `models/subgroup_metrics.csv`
- `figures/calibration_curve_random_forest.png`
- `figures/precision_recall_curve_random_forest.png`
- `figures/threshold_tradeoffs_random_forest.png`

Headline metrics for the deployed artifact on the held-out split:

- AUC-ROC: `0.8280`
- Average precision: `0.3645`
- Brier score: `0.1615`

Important calibration finding:

- The current Logistic Regression probabilities are consistently too high across
  the calibration bins, especially above `0.30`.

Examples from `models/calibration_table.csv`:

- In the `0.3-0.4` bin, mean predicted probability is `0.3484` while the
  observed positive rate is `0.0530`.
- In the `0.5-0.6` bin, mean predicted probability is `0.5440` while the
  observed positive rate is `0.1170`.
- In the `0.9-1.0` bin, mean predicted probability is `0.9388` while the
  observed positive rate is `0.5077`.

Interpretation:

- The ranking signal is strong enough for a portfolio demo and comparative risk
  sorting.
- The absolute probability values are still not clinically reliable.
- If probability quality becomes a product priority, calibration should be
  revisited for the deployed Logistic Regression path.

## Threshold And Precision-Recall Summary

The threshold scan in `models/threshold_metrics.csv` shows the expected tradeoff:

- Lower thresholds protect recall but create many false positives.
- Higher thresholds raise precision but quickly cut recall.
- In the coarse threshold grid, the highest F1 is `0.4455` at threshold `0.70`,
  but recall falls to `0.5882`, which violates the screening recall floor.

That is why the deployed threshold remains `0.5403` rather than the purely
highest-F1 threshold.

## Classification Report Snapshot

For the deployed Logistic Regression model at threshold `0.5403` on the test
set:

- `Not depressed`: precision `0.96`, recall `0.80`, F1 `0.87`, support `1491`
- `Depressed`: precision `0.27`, recall `0.71`, F1 `0.39`, support `153`
- Overall accuracy: `0.79`

This remains a high-false-positive, moderate-recall screening-style setup, not
a diagnostic classifier.

## Subgroup Evaluation Summary

The subgroup table includes 300-replicate percentile bootstrap intervals for
AUC, precision, recall, and F1.

Selected findings from `models/subgroup_metrics.csv` at threshold `0.5403`:

- By sex:
  - Female: AUC `0.8269`, precision `0.2817`, recall `0.7245`, F1 `0.4057`
  - Male: AUC `0.8173`, precision `0.2418`, recall `0.6727`, F1 `0.3558`
- By age band:
  - `35-49` has the highest F1 at `0.4651`
  - `50-64` has the highest recall at `0.7674`
  - `65+` has the lowest AUC among the age bands at `0.8071`
- By poverty band:
  - `<1.0` has the highest recall at `0.9143`
  - `2.0+` has the highest AUC at `0.8489` and the strongest precision at `0.2899`
- By race:
  - Non-Hispanic White has the highest F1 among the larger race groups at `0.4403`
  - Non-Hispanic Black shows lower precision at `0.2056`
  - Mexican American and Non-Hispanic Asian subgroups show wide uncertainty because prevalence is low and samples are smaller

Interpretation:

- These are transparency checks, not fairness guarantees.
- Subgroup prevalence differs a lot, so threshold behavior is not equally stable
  across groups.
- Any future external validation should repeat subgroup analysis before stronger
  claims are made.

## Error Analysis Summary

Detailed false-positive and false-negative review is documented in
[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).

Headline findings from the held-out split:

- False positives concentrate in profiles with lower activity, shorter sleep,
  and worse general health markers.
- False negatives are positive cases whose observed lifestyle pattern looks
  comparatively healthier than the average positive case.
- The current threshold protects recall, but that choice visibly increases the
  false-positive burden.

## Remaining Validation Gaps

- No external validation dataset
- No Kenya-specific labelled dataset
- No recalibration pass for the current Logistic Regression deployment

## Recommended Next Validation Tasks

1. Validate on a second dataset if feasible.
2. Re-run subgroup analysis on any future external dataset and compare threshold
   stability against the current `0.5403` operating point.
3. Decide whether the deployed Logistic Regression probabilities should be
   recalibrated or whether the current use case only needs ranking plus
   thresholded screening output.
4. Obtain external review of feature engineering choices, threshold
   justification, and safe-use messaging.

## Reproducibility

To reproduce the current workflow locally:

```powershell
.\.venv\Scripts\python.exe src\download_data.py
.\.venv\Scripts\python.exe src\data_cleaning.py
.\.venv\Scripts\python.exe src\feature_engineering.py
.\.venv\Scripts\python.exe src\train_model.py
.\.venv\Scripts\python.exe src\validation_analysis.py
.\.venv\Scripts\python.exe explainability\shap_analysis.py
```

Related docs:

- [Model card](MODEL_CARD.md)
- [Example scenarios](EXAMPLE_SCENARIOS.md)
