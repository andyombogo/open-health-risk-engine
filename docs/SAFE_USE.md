# Safe Use Guidance

## Intended Use

Open Health Risk Engine is appropriate for:

- portfolio demonstrations
- explainable machine learning walkthroughs
- engineering experiments with a documented inference pipeline
- teaching discussions about calibration, subgroup performance, and deployment

## Unsafe Use

Do not use this project for:

- diagnosis of depression or any other mental health condition
- crisis triage
- treatment selection
- denial of care, coverage, or benefits
- unsupervised screening in clinical settings
- any setting where an individual could be harmed by a false positive or false negative

## Why Caution Is Required

- The model is trained on public survey data, not a clinical care workflow
- Precision is limited, so many positive predictions are false positives
- Recall is limited, so many true positives are still missed
- Probability calibration is improved by post-hoc methods, but external
  validation is still missing
- Subgroup performance varies across sex, age, poverty band, and race

## Human Oversight

If you adapt this project for a research environment:

- keep a clear non-clinical disclaimer near every score output
- pair predictions with uncertainty, limitations, and subgroup caveats
- keep a human reviewer responsible for any downstream interpretation
- log model version, threshold, and timestamp for each prediction

## Communication Guidance

Use language like:

- "research demo"
- "risk estimate from survey-pattern matching"
- "not a diagnosis"

Avoid language like:

- "the patient is depressed"
- "the model confirms depression"
- "safe for clinical decisions"
