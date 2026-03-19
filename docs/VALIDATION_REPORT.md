# Validation Report

## Scope

This document summarizes the current evaluation state of the models in Open
Health Risk Engine as of March 19, 2026.

## Evaluation Setup

- Dataset: NHANES 2017-March 2020 pre-pandemic
- Outcome: `depression_binary`, defined as `PHQ-9 >= 10`
- Split: stratified 80/20 train-test split
- Cross-validation: 5-fold stratified cross-validation on the training set
- Candidate models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Reported metrics:
  - AUC-ROC
  - F1
  - Precision
  - Recall
  - Brier score on the holdout test set

## Cross-Validation Results

| Model | CV AUC-ROC | CV AUC-ROC Std | CV F1 | CV Precision | CV Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7724 | 0.0099 | 0.3136 | 0.2042 | 0.6770 |
| Random Forest | 0.7750 | 0.0095 | 0.3331 | 0.2405 | 0.5426 |
| XGBoost | 0.7449 | 0.0110 | 0.2292 | 0.2834 | 0.1934 |

## Holdout Test Results

| Model | Test AUC-ROC | Test F1 | Test Precision | Test Recall | Brier Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7811 | 0.3242 | 0.2110 | 0.6993 | 0.1889 |
| Random Forest | 0.7591 | 0.3295 | 0.2331 | 0.5621 | 0.1559 |
| XGBoost | 0.7438 | 0.2518 | 0.2800 | 0.2288 | 0.0921 |

## Current Deployment Choice

The current deployed artifact is the Random Forest model.

Reasoning for the current choice:

- It delivered the highest test F1 among the three candidate models.
- It delivered the strongest cross-validation F1 among the three candidate models.
- It supports simple feature-importance based explanation in the current app.

Tradeoff to keep in mind:

- Logistic Regression achieved the highest test AUC-ROC and recall.
- XGBoost achieved the highest precision, but recall was much lower at the current configuration.

## What The Current Metrics Mean

The current model is usable as a portfolio demo, but it still needs validation
work before stronger claims are appropriate.

Important observations:

- The problem is imbalanced, with the positive class representing a minority of cases.
- Random Forest test precision is `0.2331`, which means many positive predictions will be false positives.
- Random Forest test recall is `0.5621`, so a substantial share of true positives is still missed.
- Random Forest test Brier score is `0.1559`, but calibration plots and threshold analysis are not yet documented.

## Classification Report Snapshot

For the deployed Random Forest on the test set:

- `Not depressed`: precision `0.95`, recall `0.81`, F1 `0.87`, support `1491`
- `Depressed`: precision `0.23`, recall `0.56`, F1 `0.33`, support `153`
- Overall accuracy: `0.79`

## Major Validation Gaps

- No documented calibration curve or calibration table
- No threshold tuning study
- No subgroup performance analysis by sex, age, or poverty band
- No external validation dataset
- No error analysis report for false positives and false negatives

## Product-Relevant Limitation

The public calculator currently does not expose race/ethnicity in the UI and
passes `race_eth = 3` internally. That means the deployed calculator is not
fully personalizing that dimension today and should be framed accordingly.

## Recommended Next Validation Tasks

1. Add calibration plots and expected calibration error style summaries.
2. Compare multiple operating thresholds, not only the default classifier threshold.
3. Add subgroup evaluation by sex, age band, and poverty band.
4. Add a false-positive and false-negative review section with example patterns.
5. Reassess the deployed model choice after thresholding and calibration.

## Reproducibility

To reproduce the current training workflow locally:

```powershell
.\.venv\Scripts\python.exe src\download_data.py
.\.venv\Scripts\python.exe src\data_cleaning.py
.\.venv\Scripts\python.exe src\feature_engineering.py
.\.venv\Scripts\python.exe src\train_model.py
.\.venv\Scripts\python.exe explainability\shap_analysis.py
```

Related docs:

- [Model card](MODEL_CARD.md)
- [Example scenarios](EXAMPLE_SCENARIOS.md)
