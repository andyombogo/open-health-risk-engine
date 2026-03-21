# Fairness And Subgroup Review

## Scope

This document reviews subgroup behavior for the current Random Forest deployment
using the internal NHANES held-out test split. It is a transparency review, not
a fairness certification.

Primary artifacts:

- `models/subgroup_metrics.csv`
- `models/error_rate_by_subgroup.csv`

## Important Framing

- This is an observational survey model, not a clinically validated decision tool
- Demographic variables are proxy variables and should not be interpreted as
  causal explanations
- Small subgroup sizes make some estimates unstable
- Bootstrap confidence intervals are now reported for AUC, precision, recall,
  and F1 in `models/subgroup_metrics.csv`

## Summary Findings

### Sex

- Female participants have higher recall (`0.6020`) and higher precision
  (`0.2599`) than male participants
- Female recall bootstrap interval: `0.5107-0.7051`
- Male recall bootstrap interval: `0.3708-0.6149`
- Female participants also have a higher false positive rate (`0.2330`)
- Male participants show a higher false negative rate (`0.5091`)

Interpretation:

- The model is more likely to overcall risk for female participants
- The model is more likely to miss true positives for male participants

### Age Band

- `50-64` has the highest recall (`0.7209`) and highest F1 (`0.3713`)
- `50-64` recall bootstrap interval: `0.5807-0.8536`
- `18-34` has the highest false negative rate (`0.5429`)
- `65+` has the lowest AUC among age bands (`0.7208`)

Interpretation:

- Performance is not evenly distributed across age bands
- Younger and older groups appear easier for the model to miss than the
  `50-64` group

### Poverty Band

- `<1.0` poverty ratio has the highest recall (`0.7429`)
- `<1.0` recall bootstrap interval: `0.5810-0.8857`
- `<1.0` also has the highest false positive rate (`0.3537`)
- `2.0+` has the lowest false positive rate (`0.1243`) but the highest false
  negative rate (`0.5375`)

Interpretation:

- The model is more sensitive, but also more aggressive, in lower-income groups
- Higher-income groups are more likely to be missed when they are true positives

### Race / Ethnicity

- Non-Hispanic White has the strongest precision (`0.3000`) among the larger
  race groups
- Non-Hispanic Asian has the lowest false positive rate (`0.0765`) but a very
  high false negative rate (`0.8000`)
- Non-Hispanic Asian recall bootstrap interval: `0.0000-0.5000`
- Other Hispanic and Other / Multiracial groups show comparatively high false
  positive rates (`0.2609` and `0.2742`)

Interpretation:

- Subgroup error patterns differ by race and ethnicity
- The Non-Hispanic Asian estimate is especially unstable because the positive
  sample count is only `10`

## What This Means

- The project now documents subgroup disparities instead of hiding them behind a
  single headline metric
- These results support caution around any individual-level interpretation
- The current analysis does not establish fairness, only observed differences on
  one internal test split

## Recommended Next Steps

1. Re-run subgroup review on any future external validation dataset
2. Compare calibrated-threshold behavior across the same subgroups
3. Assess whether a reduced-feature model changes subgroup disparities
