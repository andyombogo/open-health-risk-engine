# Model Card

## Model Identity

- Project: Open Health Risk Engine
- Model artifact: `models/best_model.joblib`
- Current deployed model family: Logistic Regression selected by recall-protected tuned F1
- Current model interface: `src/predict_risk.py`
- Current public UI: `dashboard/live_app.py`
- Document date: March 25, 2026

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
- smoking
- healthcare access and self-rated health
- interaction terms

The current engineered feature set contains 63 model features.

## Outputs

`RiskPredictor.predict()` returns:

- `risk_score`: probability from 0 to 1
- `risk_label`: one of `Minimal risk`, `Low risk`, `Moderate risk`, `High risk`, `Very high risk`
- `risk_color`: UI display color token
- `phq9_estimate`: UI-oriented heuristic equal to `risk_score * 27`
- `decision_threshold`: binary operating threshold loaded from `models/optimal_threshold.json` at inference time
- `above_decision_threshold`: whether `risk_score >= decision_threshold`
- `top_factors`: top feature signals from tree importances or per-prediction linear contributions

Important notes:

- `phq9_estimate` is not a calibrated predicted PHQ-9 score.
- `top_factors` are directional model signals, not causal effects.
- `above_decision_threshold` is a demo operating-point indicator, not a clinical recommendation.

## Probability Bands Used In The UI

- `0.0 <= risk_score < 0.2`: Minimal risk
- `0.2 <= risk_score < 0.4`: Low risk
- `0.4 <= risk_score < 0.6`: Moderate risk
- `0.6 <= risk_score < 0.8`: High risk
- `0.8 <= risk_score <= 1.0`: Very high risk

## Class Imbalance Strategy

NHANES is imbalanced for the binary target `PHQ-9 >= 10` because respondents in
the depressed class make up a minority of the survey sample, which means a
model can look acceptable on overall accuracy while still underperforming on
the cases that matter most for screening-oriented use.

- SMOTE is applied inside the Random Forest training pipeline to oversample the minority class on training folds only.
- The Random Forest search uses heavier positive-class weights (`{0:1,1:4}` to `{0:1,1:6}`) to penalize missed depressed cases more heavily than false alarms.
- Threshold tuning is applied after training by selecting the best F1 among thresholds that keep recall at or above 0.70 on the held-out precision-recall curve.

| Stage | AUC | F1 | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| Before imbalance fix | 0.76 | 0.33 | 0.56 | 0.23 |
| After expanded features + tuned threshold | 0.83 | 0.39 | 0.71 | 0.27 |

## Performance Snapshot

Latest `models/best_model.joblib` performance from the repo evaluation artifacts:

| Metric | Latest best model |
| --- | ---: |
| 5-fold CV AUC-ROC (mean) | 0.8348 |
| 5-fold CV F1 (mean) | 0.3720 |
| 5-fold CV Precision (mean) | 0.2478 |
| 5-fold CV Recall (mean) | 0.7475 |
| Test AUC-ROC | 0.8280 |
| Test F1 | 0.3696 |
| Test Precision | 0.2472 |
| Test Recall | 0.7320 |
| Test Brier Score | 0.1615 |
| Tuned-threshold Test F1 | 0.3871 |
| Tuned threshold | 0.5403 |

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
- Subgroup performance evaluation is now documented for sex, age band, poverty band, and race on the internal held-out test split only, including bootstrap confidence intervals for AUC, precision, recall, and F1.
- External validation on a second dataset is not yet documented.
- The public calculator now exposes broad NHANES race/ethnicity categories, but these are demographic proxy variables rather than causal explanations.
- The app converts probability to a PHQ-9-style estimate with a simple linear scaling for UI communication only. That value should not be interpreted as a clinically validated PHQ-9 prediction.

## Ethical And Safe Use Notes

- Use this project to discuss ML design, explainability, deployment, and validation tradeoffs.
- Do not use it to make clinical decisions about an individual.
- Do not present the score as a diagnosis or a replacement for professional assessment.

## Related Docs

- [API guide](API.md)
- [Error analysis](ERROR_ANALYSIS.md)
- [Fairness review](FAIRNESS_REVIEW.md)
- [Safe-use guidance](SAFE_USE.md)
- [Validation report](VALIDATION_REPORT.md)
- [Release notes](RELEASE_NOTES.md)
- [Example scenarios](EXAMPLE_SCENARIOS.md)
- [Roadmap](../ROADMAP.md)

## Calibration Note

- A separate post-hoc calibration experiment for the Random Forest path is still
  documented in [VALIDATION_REPORT.md](VALIDATION_REPORT.md).
- The default inference path in `src/predict_risk.py` now loads
  `models/best_model.joblib` and applies the tuned threshold saved in
  `models/optimal_threshold.json`.
