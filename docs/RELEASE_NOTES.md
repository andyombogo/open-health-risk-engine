# Release Notes And Retraining Checklist

## 2026-03-25 - v0.3.1

### Highlights

- added explicit operating-threshold metadata to prediction responses and health checks
- updated the Streamlit calculator copy to explain the tuned screening threshold
- aligned API and validation docs with the current deployment path

## 2026-03-21 - v0.3.0

### Highlights

- added a README architecture diagram and calculator walkthrough image
- added friendlier loading and missing-artifact handling in the deployed app
- added a FastAPI wrapper with request validation, OpenAPI docs, API-key support,
  rate limiting, and request logging
- added fairness review, safe-use guidance, release notes, and API documentation
- added deployment-time runtime artifact verification for Docker and Render

### Model And Interface Status

- active model artifact: `models/best_model.joblib`
- feature columns artifact: `models/feature_cols.joblib`
- public UI entrypoint: `app.py`
- API entrypoint: `src/api.py`

## Retraining Checklist

Run this checklist before replacing the deployed model:

1. Regenerate raw and processed data from the documented pipeline
2. Train the candidate models and save updated artifacts into `models/`
3. Re-run validation, calibration, subgroup review, and error analysis scripts
4. Review changes in precision, recall, calibration, and subgroup behavior
5. Confirm the new `models/*.joblib` artifacts are present and load correctly
6. Update `docs/MODEL_CARD.md` and `docs/VALIDATION_REPORT.md`
7. Add a dated release note entry summarizing what changed
8. Run the test suite and a manual smoke test of the Streamlit app and API
