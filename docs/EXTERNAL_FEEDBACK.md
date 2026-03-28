# External Feedback Log

## Purpose

This file tracks one structured round of external feedback for Open Health Risk
Engine. The goal is not formal peer review. The goal is to capture how other
people experience the live app, documentation, validation story, and safety
messaging.

Use this log to close the roadmap item about collecting one round of external
feedback and summarizing what changed.

## What Counts As External Feedback

Useful reviewers include:

- a software engineer reviewing repo clarity, setup, API usability, and deployment polish
- a data scientist or ML peer reviewing feature choices, metrics, and calibration language
- a public health, clinical, or health-informatics peer reviewing claims, limitations, and safe-use wording
- a product-minded reviewer checking whether the project is easy to understand in under 60 seconds

Feedback can come from:

- GitHub issues or PR comments
- email or chat notes
- a short live walkthrough
- a document with screenshots and bullet feedback

## Reviewer Packet

When asking someone to review the project, send:

- the repository link
- the live app link
- the README
- the model card
- the validation report
- the checklist in `docs/REVIEWER_CHECKLIST.md`

If they want to respond through GitHub, point them to the issue template at
`.github/ISSUE_TEMPLATE/external-feedback.md`.

## Suggested Ask

Use a tight prompt so reviewers do not have to guess what kind of help you need:

1. Open the live app and README.
2. Spend 5 to 10 minutes trying to understand what the project does.
3. Note anything confusing, weak, misleading, or especially strong.
4. Answer the checklist sections that match your background.

## Feedback Questions

- Could you understand the project goal in under 60 seconds?
- Did the app feel credible, clear, and safe in how it explained limitations?
- Was anything confusing in the risk score, PHQ-9-equivalent wording, or threshold language?
- If you are technical, could you tell how to run or integrate the project?
- What is the single highest-impact improvement before sharing this more widely?

## Round 1 Log

| Date | Reviewer | Background | Review mode | Main feedback | Action taken | Status |
| --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD | Name or initials | SWE / DS / Clinical / Product | Call / GitHub / Message | Add notes here | Add follow-up here | Open |

## Changes Made From Feedback

- None yet

## Deferred Or Rejected Suggestions

- None yet

## Completion Criteria

You can mark the roadmap item complete when:

- at least 2 to 3 external people reviewed the project
- feedback is summarized here
- resulting changes are documented here
- major accepted changes are reflected in the README, app copy, docs, or issues
