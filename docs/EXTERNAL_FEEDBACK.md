# External Feedback Log

## Purpose

This file tracks one structured round of external feedback for Open Health Risk
Engine. The goal is not formal journal peer review. The goal is to capture how
real reviewers experience the demo, docs, validation messaging, and developer
usability.

Use this log to close the roadmap item:

- "Collect one round of external feedback and summarize what changed"

## What Counts As External Feedback

Useful reviewers include:

- a software engineer who can judge repo clarity, setup, API usability, and deployment polish
- a data scientist or ML peer who can judge feature engineering, metrics, and calibration messaging
- a public health, clinical, or health-informatics peer who can judge claims, caveats, and safe-use wording
- a product-minded friend or colleague who can judge whether the demo is easy to understand in under a minute

They do not need to review through GitHub. Feedback can come from:

- GitHub issues or PR comments
- email or WhatsApp notes
- a short Zoom or in-person walkthrough
- a message thread with screenshots and bullet feedback

If the reviewer prefers GitHub, use the issue template at
`.github/ISSUE_TEMPLATE/external-feedback.md`.

## Reviewer Packet

Send each reviewer:

- repo link
- live app link
- README
- model card
- validation report
- the reviewer checklist in `docs/REVIEWER_CHECKLIST.md`

## Suggested Ask

Ask for 15 to 20 minutes and give a narrow prompt:

1. Open the live app and README.
2. Spend 5 minutes trying to understand what the project does.
3. Note anything confusing, weak, misleading, or impressive.
4. Answer the checklist questions in the review area that best fits your background.

## Feedback Questions To Ask

- Could you understand the project goal in under 60 seconds?
- Did the app feel credible, clear, and safe in how it explains limitations?
- Was anything confusing in the score, calibration, or PHQ-9-equivalent wording?
- If you are technical, could you tell how to run or integrate the project?
- What is the one highest-impact improvement you would make before sharing this more widely?

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

- at least 2 to 3 external people reviewed it
- feedback is summarized here
- resulting changes are documented in this file
- major accepted changes are reflected in README, app copy, docs, or issues
