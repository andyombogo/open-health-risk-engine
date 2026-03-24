"""Phase 4 NLP research-track utilities."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_baseline_pipeline",
    "extract_keyword_features",
    "extract_section_map",
    "normalize_note_text",
    "prepare_note_dataframe",
    "train_baseline_text_model",
]


def __getattr__(name: str):
    if name in {"build_baseline_pipeline", "train_baseline_text_model"}:
        module = import_module("src.nlp.baseline_pipeline")
        return getattr(module, name)
    if name in {
        "extract_keyword_features",
        "extract_section_map",
        "normalize_note_text",
        "prepare_note_dataframe",
    }:
        module = import_module("src.nlp.preprocessing")
        return getattr(module, name)
    raise AttributeError(f"module 'src.nlp' has no attribute {name!r}")
