import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nlp.baseline_pipeline import train_baseline_text_model
from src.nlp.preprocessing import (
    extract_keyword_features,
    extract_section_map,
    normalize_note_text,
    prepare_note_dataframe,
)


def test_normalize_note_text_cleans_whitespace_and_deid_spans():
    text = "HISTORY:\\n[**Patient Name**] has poor sleep.\\n\\nASSESSMENT:\\nDoing better."
    normalized = normalize_note_text(text)

    assert "<deid>" in normalized
    assert "\n\n\n" not in normalized
    assert normalized.startswith("HISTORY:")


def test_extract_section_map_reads_uppercase_headers():
    text = (
        "HISTORY:\nPoor sleep and low mood.\n\n"
        "ASSESSMENT:\nSymptoms improving.\n\n"
        "PLAN:\nFollow up in two weeks."
    )
    sections = extract_section_map(text)

    assert sections["HISTORY"] == "Poor sleep and low mood."
    assert sections["ASSESSMENT"] == "Symptoms improving."
    assert sections["PLAN"] == "Follow up in two weeks."


def test_extract_keyword_features_counts_medications_and_symptoms():
    text = (
        "HISTORY:\nSertraline started recently. Patient reports fatigue and hopeless feelings.\n"
        "ASSESSMENT:\nPoor sleep remains a problem."
    )
    features = extract_keyword_features(text)

    assert features["antidepressant_mention_count"] == 1
    assert features["fatigue_mention_count"] == 1
    assert features["hopelessness_mention_count"] == 1
    assert features["sleep_issue_mention_count"] == 1
    assert features["symptom_mention_count"] == 3


def test_prepare_note_dataframe_adds_structured_features():
    frame = pd.DataFrame(
        {
            "note_text": [
                "HISTORY:\nFatigue and insomnia.\nASSESSMENT:\nSertraline started.\nPLAN:\nFollow up."
            ],
            "label": [1],
        }
    )
    prepared = prepare_note_dataframe(frame)

    assert prepared.loc[0, "section_count"] == 3
    assert prepared.loc[0, "has_history_section"] == 1
    assert prepared.loc[0, "has_plan_section"] == 1
    assert prepared.loc[0, "has_antidepressant_mention"] == 1
    assert prepared.loc[0, "has_sleep_issue_mention"] == 1


def test_train_baseline_text_model_returns_metrics_and_split_sizes():
    frame = pd.DataFrame(
        {
            "note_text": [
                "HISTORY:\nLow mood and insomnia.\nASSESSMENT:\nDepression likely.",
                "HISTORY:\nSleeping well and active.\nASSESSMENT:\nStable mood.",
                "HISTORY:\nHopelessness and fatigue.\nASSESSMENT:\nNeeds closer follow-up.",
                "HISTORY:\nGood energy and regular exercise.\nASSESSMENT:\nDoing well.",
                "HISTORY:\nPoor appetite and poor sleep.\nASSESSMENT:\nSymptoms worsening.",
                "HISTORY:\nNo mood complaints.\nASSESSMENT:\nStable.",
                "HISTORY:\nAnxiety, low mood, and insomnia.\nASSESSMENT:\nHigh risk.",
                "HISTORY:\nMood improved and sleeping seven hours.\nASSESSMENT:\nLow concern.",
            ],
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    result = train_baseline_text_model(frame)

    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert "symptom_mention_count" in result["numeric_feature_columns"]
    for metric_name, metric_value in result["metrics"].items():
        assert 0.0 <= metric_value <= 1.0, metric_name
