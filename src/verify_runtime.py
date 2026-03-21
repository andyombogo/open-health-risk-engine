"""Verify that deployment-time artifacts required for inference are present."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_ARTIFACTS = [
    ROOT / "models" / "best_model.joblib",
    ROOT / "models" / "feature_cols.joblib",
]


def find_missing_artifacts() -> list[Path]:
    """Return any required inference artifacts that are missing from the repo."""
    return [path for path in REQUIRED_ARTIFACTS if not path.exists()]


def main() -> int:
    missing = find_missing_artifacts()
    if missing:
        print("Missing runtime artifacts:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("Runtime artifacts verified:")
    for path in REQUIRED_ARTIFACTS:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
