# MIMIC Access Checklist

## Goal

Phase 4 depends on access to de-identified clinical notes before any real
note-level modeling can begin. This file keeps the access work visible without
mixing it into the main NHANES demo.

## Access Checklist

1. Create or confirm a PhysioNet account
2. Review the current MIMIC-IV access requirements directly in PhysioNet
3. Complete the required training and data-use steps
4. Submit the credentialing request
5. Store approval dates and renewal dates in project notes
6. Do not place credentialed data in this public repository

## Repo Rules

- Keep all credentialed data outside this repo
- Keep any local paths or secrets out of committed config files
- Use `.env` or local-only config for private paths if needed later
- Do not merge note samples into docs or tests unless they are synthetic

## What We Can Do Before Access Is Granted

- build text preprocessing utilities
- define note schema assumptions
- set up a baseline NLP training pipeline
- document evaluation criteria and safe-use boundaries

## What Waits On Access

- discharge-summary experiments
- medication mention extraction on real notes
- BioBERT or ClinicalBERT fine-tuning
- structured-plus-text comparison on real data
