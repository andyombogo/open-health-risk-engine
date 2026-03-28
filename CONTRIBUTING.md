# Contributing

Thanks for contributing to Open Health Risk Engine.

This project is both a portfolio demo and a foundation for larger clinical data
science work, so contributions should improve clarity, reproducibility, and
honesty just as much as raw model performance.

## What To Work On

Good contribution areas include:

- Documentation and reproducibility improvements
- Validation, calibration, and error analysis
- UI polish and accessibility fixes
- API and developer-experience improvements
- Tests, CI, and deployment reliability

Before starting large work, open an issue or discussion so the direction stays
aligned with the roadmap.

## External Feedback And Review

Non-code review is useful on this project too. If you are reviewing the live
demo, documentation, or safety language:

- Use the prompts in `docs/REVIEWER_CHECKLIST.md`
- Open a GitHub issue with the external feedback template when possible
- Summarize accepted changes in `docs/EXTERNAL_FEEDBACK.md`
- Focus on clarity, trustworthiness, and whether the project is understandable
  in under a minute

## Local Setup

```powershell
py -3 -m venv .venv
py -3 -m pip --python .\.venv\Scripts\python.exe install -r requirements.txt
```

## Common Commands

Run the full pipeline:

```powershell
.\.venv\Scripts\python.exe src\download_data.py
.\.venv\Scripts\python.exe src\data_cleaning.py
.\.venv\Scripts\python.exe src\feature_engineering.py
.\.venv\Scripts\python.exe src\train_model.py
.\.venv\Scripts\python.exe explainability\shap_analysis.py
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Launch the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Contribution Guidelines

- Keep changes focused and explain the user-facing impact in your PR.
- Add or update tests when behavior changes.
- Update docs when inputs, outputs, metrics, or deployment behavior change.
- Do not commit secrets, tokens, or raw restricted health data.
- Do not make unsupported clinical claims. If a change affects interpretation,
  update the model card or validation docs too.
- Prefer simple, auditable improvements over complexity that is hard to explain.

## Pull Request Checklist

- [ ] Code runs locally
- [ ] Tests pass locally
- [ ] Documentation is updated where needed
- [ ] No credentials or sensitive data were added
- [ ] Clinical and ethical limitations remain accurately described

## Review Lens

Reviewers should prioritize:

- Behavioral regressions
- Evaluation quality and metric interpretation
- Reproducibility
- Deployment reliability
- Documentation clarity

## Questions

Use the roadmap in [ROADMAP.md](ROADMAP.md) to understand current priorities,
especially the portfolio-hardening and validation work.
