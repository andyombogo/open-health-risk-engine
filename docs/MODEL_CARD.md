# Model Card

## Model Identity

- Project: Open Health Risk Engine
- Model artifact: `models/best_model.joblib`
- Current deployed model family: Random Forest
- Current model interface: `src/predict_risk.py`
- Current public UI: `dashboard/live_app.py`
- Document date: March 19, 2026

## Summary

This model estimates the probability that a respondent falls into the binary
target `depression_binary`, defined in this project as `PHQ-9 >= 10`.

It is designed for:

- portfolio demonstration
- explainable ML workflow illustration
- experimentation with public health survey features

It is not designed for:

- diagnosis
- treatment decisions
- crisis triage
- standalone screening in clinical care

## Training Data

- Source: NHANES 2017-March 2020 pre-pandemic public survey data
- Label: `depression_binary = 1` when `PHQ-9 >= 10`
- Problem type: binary classification
- Train/test split: stratified 80/20 holdout
- Validation: 5-fold stratified cross-validation on the training split

## Inputs

The live calculator currently collects these direct user-facing inputs:

- age
- sex
- poverty-income ratio
- weekly physical activity
- sleep hours
- sleep trouble
- BMI
- drinks per week
- education

These are expanded into engineered features covering:

- demographics
- physical activity
- sleep
- BMI
- alcohol use
- interaction terms

The current engineered feature set contains 30 model features.

## Outputs

`RiskPredictor.predict()` returns:

- `risk_score`: probability from 0 to 1
- `risk_label`: one of `Minimal risk`, `Low risk`, `Moderate risk`, `High risk`, `Very high risk`
- `risk_color`: UI display color token
- `phq9_estimate`: UI-oriented heuristic equal to `risk_score * 27`
- `top_factors`: top feature-importance entries from the trained Random Forest

Important notes:

- `phq9_estimate` is not a calibrated predicted PHQ-9 score.
- `top_factors` come from model feature importances and do not represent causal effects.

## Probability Bands Used In The UI

- `0.0 <= risk_score < 0.2`: Minimal risk
- `0.2 <= risk_score < 0.4`: Low risk
- `0.4 <= risk_score < 0.6`: Moderate risk
- `0.6 <= risk_score < 0.8`: High risk
- `0.8 <= risk_score <= 1.0`: Very high risk

## Performance Snapshot

Current deployed model performance from the repo evaluation artifacts:

| Metric | Random Forest |
| --- | ---: |
| 5-fold CV AUC-ROC (mean) | 0.7750 |
| 5-fold CV F1 (mean) | 0.3331 |
| 5-fold CV Precision (mean) | 0.2405 |
| 5-fold CV Recall (mean) | 0.5426 |
| Test AUC-ROC | 0.7591 |
| Test F1 | 0.3295 |
| Test Precision | 0.2331 |
| Test Recall | 0.5621 |
| Test Brier Score | 0.1559 |

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for model comparison and
interpretation.

Additional validation artifacts now tracked in the repo:

- `models/calibration_table.csv`
- `models/threshold_metrics.csv`
- `models/subgroup_metrics.csv`
- `figures/calibration_curve_random_forest.png`
- `figures/precision_recall_curve_random_forest.png`
- `figures/threshold_tradeoffs_random_forest.png`

## Known Limitations

- This is a public-survey model, not a clinical prediction model validated for care delivery.
- Positive-class precision is currently low, so false positives are expected.
- Calibration and threshold analysis are now documented for the internal held-out test split, but external calibration is not yet available.
- Subgroup performance evaluation is now documented for sex, age band, poverty band, and race on the internal held-out test split only.
- External validation on a second dataset is not yet documented.
- The public calculator now exposes broad NHANES race/ethnicity categories, but these are demographic proxy variables rather than causal explanations.
- The app converts probability to a PHQ-9-style estimate with a simple linear scaling for UI communication only. That value should not be interpreted as a clinically validated PHQ-9 prediction.

## Ethical And Safe Use Notes

- Use this project to discuss ML design, explainability, deployment, and validation tradeoffs.
- Do not use it to make clinical decisions about an individual.
- Do not present the score as a diagnosis or a replacement for professional assessment.

## Related Docs

- [Error analysis](ERROR_ANALYSIS.md)
- [Validation report](VALIDATION_REPORT.md)
- [Example scenarios](EXAMPLE_SCENARIOS.md)
- [Roadmap](../ROADMAP.md)

## Calibration Note

- Post-hoc calibration improves probability quality significantly (Brier from
  `0.1559` to `0.0772` with sigmoid calibration).
- The calibrated model favors a lower decision threshold (~`0.20`) if balancing
  precision and recall on the current test split.
