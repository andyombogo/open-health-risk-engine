# Phase 4 Plan

## Positioning

Phase 4 is a parallel research track inside the same repository. The current
NHANES lifestyle calculator remains the stable public artifact. NLP work should
not change the deployed app or public claims until note-based experiments are
real, tested, and documented.

## Immediate Objective

Stand up the minimum code and documentation needed to begin note-based
experiments once credentialed access is available.

## Phase 4 Work Order

1. Secure access to a note dataset through the approved access path
2. Build note cleaning and section extraction utilities
3. Train a lightweight TF-IDF plus Logistic Regression baseline
4. Define task labels and evaluation criteria
5. Add transformer experiments only after the baseline is working
6. Compare text-only and structured-plus-text setups later

## Initial Repo Scope

The new `src/nlp/` package should cover:

- note normalization
- section extraction
- keyword-level feature extraction for medications and core symptoms
- baseline text-model training
- isolated tests that do not depend on credentialed data

The repo now also includes:

- a synthetic CSV in `data/synthetic/phase4_note_labels.csv`
- a demo notebook in `notebooks/phase4_nlp_baseline_demo.ipynb`

## Guardrails

- keep the current deployed app unchanged
- keep Phase 4 claims clearly labeled as research work
- do not imply clinical-note support in the public app yet
- keep synthetic or toy text in tests only

## Success Criteria For This Track

- the repo has a clean NLP package and docs
- the baseline pipeline can train from a generic CSV with note text and labels
- the work is isolated enough that it does not destabilize the main project
