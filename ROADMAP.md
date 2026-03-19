# Project Roadmap

This document tracks what is done, what is next, and what will most improve the
project as a reliable public demo, a strong portfolio piece, and a credible
technical foundation.

---

## Success Criteria

For this project to feel successful to employers, collaborators, and reviewers,
it should:

- Open from the README with one click and load reliably
- Explain the problem, data, model, and limitations in under 60 seconds
- Show a polished live calculator plus a reproducible ML pipeline
- Include honest evaluation, calibration, and subgroup checks
- Be easy to extend through an API and clear project structure
- Be documented well enough that another engineer can run it locally

---

## Phase 1 - NHANES Lifestyle Engine (foundation complete)
**Goal:** Predict PHQ-9 depression severity from lifestyle and demographic data.

- [x] NHANES 2017-2020 data pipeline
- [x] Epidemiologically grounded feature engineering
- [x] Random Forest and XGBoost models with cross-validation
- [x] SHAP explainability (global and local)
- [x] Streamlit interactive dashboard
- [x] Unit tests and CI workflow
- [x] Hugging Face Spaces deployment
- [x] Render web deployment
- [ ] Kaggle notebook version for reach and visibility
- [ ] External review of feature engineering and modeling decisions

---

## Phase 1.5 - Portfolio and product hardening (highest priority)
**Goal:** Make the current project stronger as a public demo before adding new
research tracks.

- [x] Add a model card covering intended use, limitations, metrics, and the non-clinical disclaimer
- [x] Add calibration, threshold tuning, and precision-recall analysis for the deployed model
- [x] Add subgroup evaluation by sex, age band, and poverty band
- [ ] Add a short architecture diagram and one demo GIF or screenshot walkthrough to README
- [x] Add example user scenarios with expected outputs
- [x] Add `LICENSE` and `CONTRIBUTING.md`
- [ ] Fix repo hygiene issues such as encoding cleanup, `.gitignore` warnings, and doc consistency
- [ ] Add friendlier loading, empty-state, and failure-state handling in the live app
- [ ] Add mobile and accessibility polish for the calculator UI
- [ ] Collect one round of external feedback and summarize what changed

---

## Phase 2 - API and developer usability
**Goal:** Make the scoring engine reusable by other apps and easier for other
engineers to integrate.

- [ ] FastAPI wrapper around `predict_risk.py`
- [ ] Request and response schema validation
- [ ] OpenAPI / Swagger documentation
- [ ] Authentication with API keys
- [ ] Rate limiting and request logging
- [ ] Example `curl` and Postman collection
- [ ] Example integration stub for another web app

---

## Phase 3 - Validation and clinical credibility
**Goal:** Strengthen evidence and transparency before any real clinical
integration claims.

- [ ] External review of feature engineering decisions
- [ ] Error analysis for false positives and false negatives
- [ ] Fairness and subgroup performance review
- [ ] External validation on a second dataset if feasible
- [ ] Safe-use and unsafe-use guidance
- [ ] Versioned model release notes and retraining checklist

---

## Phase 4 - NLP on Clinical Notes
**Goal:** Extract mental health signals from free-text clinical notes as a
separate research track.

- [ ] MIMIC-IV credentialing and access through PhysioNet
- [ ] NLP feature extraction from discharge summaries
- [ ] Medication mention extraction
- [ ] Symptom language extraction for fatigue, hopelessness, and sleep issues
- [ ] Fine-tune BioBERT or ClinicalBERT for PHQ-9 related text patterns
- [ ] Integrate NLP features into the structured-data model
- [ ] Compare NLP-enhanced performance vs. structured-data-only performance

---

## Phase 5 - Longitudinal Risk Modeling
**Goal:** Predict future PHQ-9 trajectory, not only current risk.

- [ ] Repeated-measures modeling with mixed effects or sequence models
- [ ] Time-to-crisis survival analysis
- [ ] Missed appointment patterns as a risk signal
- [ ] Medication refill gaps as a risk signal

---

## Phase 6 - Population-Level Dashboard
**Goal:** Support county or district-level mental health risk surveillance.

- [ ] Aggregate individual scores into population estimates
- [ ] Geographic visualization with Leaflet or Plotly choropleths
- [ ] Time-series trend analysis
- [ ] Alerting for population-level risk spikes
- [ ] Kenya-specific pilot using KHIS or DHIS2 data

---

## Long-Term Vision

A modular, open clinical decision support platform that:
1. Works with public health survey data such as NHANES, DHS, and SARA
2. Integrates with health systems through standard APIs
3. Supports individual scoring and population-level surveillance
4. Is explainable, auditable, and transparent about limitations
5. Has been validated in at least one real-world setting

---

## Recommended Next Moves

1. Finish Phase 1.5 before starting NLP or longitudinal modeling.
2. Prioritize API usability and validation before EHR or public-health expansion.
3. Treat clinical integration claims as a later milestone, not a near-term marketing point.

---

*Contributions welcome. See README.md for how to get involved.*
