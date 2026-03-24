"""Preprocessing helpers for Phase 4 clinical-note experiments."""

from __future__ import annotations

import re

import pandas as pd

DEIDENTIFIED_SPAN_PATTERN = re.compile(r"\[\*\*.*?\*\*\]")
SECTION_HEADER_PATTERN = re.compile(r"^[A-Z][A-Z /&-]{2,}:$")

ANTIDEPRESSANT_TERMS = (
    "sertraline",
    "fluoxetine",
    "escitalopram",
    "citalopram",
    "paroxetine",
    "venlafaxine",
    "duloxetine",
    "bupropion",
    "mirtazapine",
    "trazodone",
)

SYMPTOM_TERM_GROUPS = {
    "fatigue": ("fatigue", "tired", "exhausted", "low energy"),
    "hopelessness": ("hopeless", "worthless", "helpless"),
    "sleep_issue": ("insomnia", "poor sleep", "sleep trouble", "sleep disturbance"),
}

SECTION_FLAG_HEADERS = ("HISTORY", "ASSESSMENT", "PLAN")


def count_term_mentions(text: str, terms: tuple[str, ...]) -> int:
    """Count whole-term matches for a small clinical lexicon."""
    lowered = normalize_note_text(text).lower()
    if not lowered:
        return 0

    count = 0
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term.lower())}\b")
        count += len(pattern.findall(lowered))
    return count


def extract_keyword_features(text: str) -> dict[str, int]:
    """Extract lightweight keyword-count and flag features from note text."""
    medication_count = count_term_mentions(text, ANTIDEPRESSANT_TERMS)
    fatigue_count = count_term_mentions(text, SYMPTOM_TERM_GROUPS["fatigue"])
    hopelessness_count = count_term_mentions(text, SYMPTOM_TERM_GROUPS["hopelessness"])
    sleep_issue_count = count_term_mentions(text, SYMPTOM_TERM_GROUPS["sleep_issue"])
    total_symptom_mentions = fatigue_count + hopelessness_count + sleep_issue_count

    return {
        "antidepressant_mention_count": medication_count,
        "fatigue_mention_count": fatigue_count,
        "hopelessness_mention_count": hopelessness_count,
        "sleep_issue_mention_count": sleep_issue_count,
        "symptom_mention_count": total_symptom_mentions,
        "has_antidepressant_mention": int(medication_count > 0),
        "has_fatigue_mention": int(fatigue_count > 0),
        "has_hopelessness_mention": int(hopelessness_count > 0),
        "has_sleep_issue_mention": int(sleep_issue_count > 0),
    }


def normalize_note_text(text: str) -> str:
    """Normalize raw note text into a cleaner baseline form."""
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = DEIDENTIFIED_SPAN_PATTERN.sub("<deid>", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_section_map(text: str) -> dict[str, str]:
    """Extract simple all-caps note sections into a dictionary."""
    normalized = normalize_note_text(text)
    if not normalized:
        return {}

    sections: dict[str, str] = {}
    current_header = "UNSPECIFIED"
    current_lines: list[str] = []

    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_HEADER_PATTERN.fullmatch(stripped):
            if current_lines:
                sections[current_header] = " ".join(current_lines).strip()
            current_header = stripped[:-1]
            current_lines = []
            continue
        current_lines.append(stripped)

    if current_lines:
        sections[current_header] = " ".join(current_lines).strip()

    return sections


def prepare_note_dataframe(
    frame: pd.DataFrame,
    text_column: str = "note_text",
    label_column: str = "label",
) -> pd.DataFrame:
    """Validate and enrich a note-level DataFrame for text modeling."""
    missing_columns = {text_column, label_column} - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    prepared = frame[[text_column, label_column]].copy()
    prepared = prepared.dropna(subset=[text_column, label_column]).copy()
    prepared[text_column] = prepared[text_column].map(normalize_note_text)
    prepared = prepared[prepared[text_column] != ""].copy()
    prepared[label_column] = prepared[label_column].astype(int)
    prepared["section_map"] = prepared[text_column].map(extract_section_map)
    prepared["section_count"] = prepared["section_map"].map(len)
    prepared["word_count"] = prepared[text_column].map(lambda text: len(text.split()))
    for header in SECTION_FLAG_HEADERS:
        feature_name = f"has_{header.lower()}_section"
        prepared[feature_name] = prepared["section_map"].map(
            lambda sections, key=header: int(key in sections)
        )

    keyword_features = prepared[text_column].map(extract_keyword_features).apply(pd.Series)
    prepared = pd.concat([prepared, keyword_features], axis=1)
    prepared = prepared.drop(columns=["section_map"])
    return prepared.reset_index(drop=True)
