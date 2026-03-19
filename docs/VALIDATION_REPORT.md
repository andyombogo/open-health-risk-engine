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
- Random Forest test Brier score is `0.1559`.

## Calibration Summary

Internal calibration analysis is now saved in `models/calibration_table.csv` and
`figures/calibration_curve_random_forest.png`.

Key observation:

- The model overpredicts risk across every calibration bin in the held-out test set.

Examples from the calibration table:

- In the `0.5-0.6` probability bin, the mean predicted probability is `0.5441` but the observed positive rate is `0.1691`.
- In the `0.6-0.7` probability bin, the mean predicted probability is `0.6489` but the observed positive rate is `0.2324`.
- In the `0.7-0.8` probability bin, the mean predicted probability is `0.7360` but the observed positive rate is `0.3452`.

Interpretation:

- The ranking signal is useful enough for a portfolio demo, but the raw probability values are not well calibrated and should not be interpreted as clinically reliable absolute risk.

## Threshold And Precision-Recall Summary

Threshold analysis is saved in `models/threshold_metrics.csv` and
`figures/threshold_tradeoffs_random_forest.png`.
The precision-recall curve is saved in
`figures/precision_recall_curve_random_forest.png`.

Key observations:

- Average precision on the held-out test set is `0.2434`.
- At the default `0.50` threshold, precision is `0.2331`, recall is `0.5621`, and F1 is `0.3295`.
- In the scanned threshold grid, the highest F1 is `0.3333` at threshold `0.65`, with precision `0.3333` and recall `0.3333`.
- Threshold `0.55` is close behind with F1 `0.3311`, precision `0.2535`, and recall `0.4771`.

Interpretation:

- A stricter threshold improves precision somewhat, but it quickly reduces recall.
- There is no single threshold that resolves the current tradeoff; calibration and intended use still matter.

## Classification Report Snapshot

For the deployed Random Forest on the test set:

- `Not depressed`: precision `0.95`, recall `0.81`, F1 `0.87`, support `1491`
- `Depressed`: precision `0.23`, recall `0.56`, F1 `0.33`, support `153`
- Overall accuracy: `0.79`

## Subgroup Evaluation Summary

Subgroup evaluation is saved in `models/subgroup_metrics.csv`.
The current analysis covers sex, age band, poverty band, and race on the
held-out test split.

Selected findings:

- By sex:
  - Female: AUC `0.7423`, precision `0.2599`, recall `0.6020`, F1 `0.3631`
  - Male: AUC `0.7615`, precision `0.1901`, recall `0.4909`, F1 `0.2741`
- By age band:
  - `50-64` has the highest recall at `0.7209` and the highest F1 at `0.3713`
  - `65+` has the lowest AUC among the age bands at `0.7208`
- By poverty band:
  - `<1.0` has the highest recall at `0.7429`
  - `2.0+` has the highest AUC at `0.7848`
- By race:
  - Non-Hispanic White: precision `0.3000`, recall `0.5909`, F1 `0.3980`
  - Non-Hispanic Asian: AUC `0.8214`, but prevalence is low and recall is `0.2000`
  - Other / Multiracial: F1 `0.4103`, but sample size is only `76`

Interpretation:

- The subgroup results are useful for transparency, but they are not fairness guarantees.
- Group sizes and prevalence vary substantially, so some subgroup metrics are less stable than others.

## Error Analysis Summary

Detailed false-positive and false-negative review is now documented in
[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).

Headline findings from the held-out test split:

- False positives are concentrated in profiles with low activity, high sleep trouble, higher BMI, and lower poverty ratio compared with true negatives.
- False negatives are positive cases whose observed lifestyle pattern looks comparatively healthier than true positives.
- Female participants show higher false positive rates, while male participants show higher false negative rates.
- Below-poverty participants show higher false positive rates, while higher-income participants show higher false negative rates.

## Remaining Validation Gaps

- No external validation dataset
- No post-hoc probability calibration method has been applied yet
- No confidence intervals are reported for subgroup metrics

## Recommended Next Validation Tasks

1. Add calibration plots and expected calibration error style summaries.
2. Reassess the deployed model choice after thresholding and calibration.
3. Add confidence intervals or bootstrap uncertainty for subgroup metrics.
4. Validate on a second dataset if feasible.

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
