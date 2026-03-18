# Project Roadmap

This document tracks where the project has been and where it's going.
Updated as milestones are completed.

---

## Phase 1 — NHANES Lifestyle Engine (current)
**Goal:** Predict PHQ-9 depression severity from lifestyle + demographic data.

- [x] NHANES 2017–2020 data pipeline
- [x] Epidemiologically principled feature engineering
- [x] Random Forest + XGBoost models with cross-validation
- [x] SHAP explainability (global + local)
- [x] Streamlit interactive dashboard
- [x] Unit tests + CI pipeline
- [ ] Hugging Face Spaces deployment
- [ ] Kaggle notebook version (for reach/visibility)
- [ ] Peer review of feature engineering decisions

---

## Phase 2 — NLP on Clinical Notes
**Goal:** Extract mental health signals from free-text clinical notes.

- [ ] MIMIC-IV credentialing + access (free via PhysioNet)
- [ ] NLP feature extraction from discharge summaries
  - Medication mentions (antidepressants, anxiolytics)
  - Symptom language (fatigue, hopelessness, sleep issues)
  - Fine-tuned BioBERT or ClinicalBERT for PHQ-9 related language
- [ ] Integrate NLP features into Phase 1 model
- [ ] Evaluate NLP vs. structured-data-only performance

---

## Phase 3 — Longitudinal Risk Modeling
**Goal:** Predict future PHQ-9 trajectory, not just current score.

- [ ] Repeated measures modeling (mixed-effects or LSTM)
- [ ] "Time to crisis" survival analysis
- [ ] Feature: missed appointment patterns as risk signal
- [ ] Feature: medication refill gaps as risk signal

---

## Phase 4 — REST API for System Integration
**Goal:** Make the engine callable from EHR systems and third-party apps.

- [ ] FastAPI wrapper around `predict_risk.py`
- [ ] Docker container + docker-compose setup
- [ ] Authentication (API key)
- [ ] OpenAPI / Swagger documentation
- [ ] Rate limiting + logging
- [ ] Example: Epic MyChart integration stub

---

## Phase 5 — Population-Level Dashboard
**Goal:** County/district-level mental health risk surveillance.

- [ ] Aggregate individual scores to population estimates
- [ ] Geographic visualization (Leaflet.js or Plotly Choropleth)
- [ ] Time-series trend analysis
- [ ] Alerting system for population-level risk spikes
- [ ] Kenya-specific pilot: integrate KHIS / DHIS2 data

---

## Long-term vision

A modular, open clinical decision support platform that:
1. Works with public health survey data (NHANES, DHS, SARA)
2. Integrates with EHR systems via standard APIs (HL7 FHIR)
3. Provides county/district level population risk estimates
4. Is fully explainable and auditable by clinicians
5. Has been validated in at least one real-world clinical setting

---

*Contributions welcome. See README.md for how to get involved.*
