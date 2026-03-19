# Example Scenarios

These examples were generated from the local model bundle in
`models/best_model.joblib` on March 19, 2026.

They are intended to show how the current demo behaves, not to provide clinical
advice.

## Scenario Summary

| Scenario | Risk Score | Risk Label | PHQ-9 Estimate |
| --- | ---: | --- | ---: |
| Low-risk active male | 0.0652 | Minimal risk | 1.8 |
| Higher-risk sleep and poverty | 0.8587 | Very high risk | 23.2 |
| Moderate-risk sedentary | 0.5025 | Moderate risk | 13.6 |

## 1. Low-Risk Active Male

Inputs:

- Age: 30
- Sex: Male
- Poverty ratio: 3.5
- Activity: 900 MET-min/week
- Sleep: 8.0 hours
- Sleep trouble: No
- BMI: 23.0
- Drinks/week: 2
- Education: 5
- Race input used: 3

Output:

- Risk score: `0.0652`
- Risk label: `Minimal risk`
- PHQ-9 estimate: `1.8`

Top factors returned:

- `sleep_trouble`
- `poverty_ratio`
- `bmi`
- `sleep_hours`
- `age`

## 2. Higher-Risk Sleep And Poverty

Inputs:

- Age: 45
- Sex: Female
- Poverty ratio: 0.8
- Activity: 0 MET-min/week
- Sleep: 5.0 hours
- Sleep trouble: Yes
- BMI: 32.0
- Drinks/week: 14
- Education: 2
- Race input used: 4 for the local example

Output:

- Risk score: `0.8587`
- Risk label: `Very high risk`
- PHQ-9 estimate: `23.2`

Top factors returned:

- `sleep_trouble`
- `poverty_ratio`
- `bmi`
- `sleep_hours`
- `age`

## 3. Moderate-Risk Sedentary

Inputs:

- Age: 52
- Sex: Male
- Poverty ratio: 1.4
- Activity: 100 MET-min/week
- Sleep: 6.0 hours
- Sleep trouble: Yes
- BMI: 29.5
- Drinks/week: 8
- Education: 3
- Race input used: 3

Output:

- Risk score: `0.5025`
- Risk label: `Moderate risk`
- PHQ-9 estimate: `13.6`

Top factors returned:

- `sleep_trouble`
- `poverty_ratio`
- `bmi`
- `sleep_hours`
- `age`

## Notes

- These outputs can change after retraining or threshold changes.
- The public app now exposes the same broad NHANES race and ethnicity categories used in training, so live outputs can vary when that input changes.
- `PHQ-9 estimate` is a UI heuristic and not a clinically validated predicted PHQ-9 score.
