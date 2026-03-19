# Error Analysis

This document reviews false positives and false negatives for the deployed
Random Forest model on the held-out NHANES test split used throughout this
project.

Artifacts generated from this analysis:

- `models/error_outcome_summary.csv`
- `models/error_feature_deltas.csv`
- `models/error_rate_by_subgroup.csv`
- `figures/confusion_matrix_random_forest.png`

## Confusion Breakdown

At the default `0.50` threshold, the held-out test split breaks down as:

| Outcome | Count | Share of test split |
| --- | ---: | ---: |
| True negative | 1208 | 73.48% |
| False positive | 283 | 17.21% |
| False negative | 67 | 4.08% |
| True positive | 86 | 5.23% |

Equivalent error rates:

- False positive rate on actual negatives: `283 / 1491 = 18.98%`
- False negative rate on actual positives: `67 / 153 = 43.79%`

## False Positive Pattern

Compared with true negatives, false positives look much more like the model's
learned high-risk lifestyle profile:

| Feature | False positives | True negatives | Difference |
| --- | ---: | ---: | ---: |
| Female share | 0.5936 | 0.4578 | +0.1359 |
| Poverty ratio | 1.8477 | 2.8720 | -1.0243 |
| Weekly activity (MET-min) | 504.8269 | 4315.7964 | -3810.9695 |
| Sleep trouble rate | 0.8339 | 0.1068 | +0.7271 |
| BMI | 32.9653 | 28.9088 | +4.0565 |
| Education | 3.3004 | 3.7467 | -0.4463 |
| Predicted probability | 0.6264 | 0.2683 | +0.3581 |

Interpretation:

- The model strongly flags low activity, sleep trouble, higher BMI, and lower
  income as risk signals.
- Many false positives still have low PHQ-9 in the test set even though their
  lifestyle profile resembles higher-risk patterns.
- This is consistent with the calibration finding that the model overpredicts
  risk across bins.

## False Negative Pattern

Compared with true positives, false negatives are depressed respondents whose
measured lifestyle profile looks comparatively healthier to the model:

| Feature | False negatives | True positives | Difference |
| --- | ---: | ---: | ---: |
| Female share | 0.5821 | 0.6860 | -0.1040 |
| Poverty ratio | 2.9606 | 1.8362 | +1.1244 |
| Weekly activity (MET-min) | 4316.2313 | 809.6279 | +3506.6034 |
| Sleep trouble rate | 0.3582 | 0.9767 | -0.6185 |
| BMI | 30.0579 | 31.8482 | -1.7902 |
| Education | 3.5970 | 3.1628 | +0.4342 |
| Predicted probability | 0.3340 | 0.6574 | -0.3234 |

Interpretation:

- The model misses a meaningful subset of positive cases when the measured
  lifestyle variables do not look especially high-risk.
- This suggests the current feature set is probably missing important
  non-lifestyle drivers of PHQ-9 severity such as psychosocial stressors,
  prior history, or factors not present in this public survey slice.

## Error Rates By Subgroup

Selected subgroup findings from `models/error_rate_by_subgroup.csv`:

### Sex

- Female: false positive rate `0.2330`, false negative rate `0.3980`
- Male: false positive rate `0.1494`, false negative rate `0.5091`

Interpretation:

- The model is more likely to overcall risk for female participants.
- The model is more likely to miss positive cases for male participants.

### Age Band

- `18-34`: false positive rate `0.1395`, false negative rate `0.5429`
- `35-49`: false positive rate `0.2092`, false negative rate `0.4500`
- `50-64`: false positive rate `0.2403`, false negative rate `0.2791`
- `65+`: false positive rate `0.1777`, false negative rate `0.5143`

Interpretation:

- The model misses more positives in younger and older bands than in `50-64`.
- The `50-64` band has the strongest positive-case capture, but also a higher
  false positive rate.

### Poverty Band

- `<1.0`: false positive rate `0.3537`, false negative rate `0.2571`
- `1.0-1.99`: false positive rate `0.2564`, false negative rate `0.3947`
- `2.0+`: false positive rate `0.1243`, false negative rate `0.5375`

Interpretation:

- The model is much more likely to overcall risk among participants below the
  poverty line.
- It is much more likely to miss positives among higher-income participants.

### Race / Ethnicity

- Mexican American: false positive rate `0.1847`, false negative rate `0.2727`
- Other Hispanic: false positive rate `0.2609`, false negative rate `0.3125`
- Non-Hispanic White: false positive rate `0.1802`, false negative rate `0.4091`
- Non-Hispanic Black: false positive rate `0.2194`, false negative rate `0.5000`
- Non-Hispanic Asian: false positive rate `0.0765`, false negative rate `0.8000`
- Other / Multiracial: false positive rate `0.2742`, false negative rate `0.4286`

Interpretation:

- The subgroup pattern is not uniform across race and ethnicity.
- The `Non-Hispanic Asian` false negative rate is especially high, but the
  positive sample count is only `10`, so that estimate is unstable.
- `Other Hispanic` and `Other / Multiracial` groups show comparatively high
  false positive rates, but sample size should still be considered.

## What This Means For The Project

This is a strong portfolio finding because it shows the project is not only
reporting a single headline AUC or F1 score. It is examining where the model
fails and what those failures look like.

The next technical move after this analysis should be:

1. post-hoc calibration
2. threshold reassessment
3. confidence intervals for subgroup metrics
4. external validation if a second dataset becomes feasible
