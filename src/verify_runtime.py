"""Verify that deployment-time artifacts required for inference are present."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict_risk import resolve_model_path


def get_required_artifacts(model_path: Path | None = None) -> list[Path]:
    """Return the inference artifacts required by the deployed predictor."""
    return [
        resolve_model_path(model_path),
        ROOT / "models" / "feature_cols.joblib",
    ]


REQUIRED_ARTIFACTS = get_required_artifacts()


def find_missing_artifacts(model_path: Path | None = None) -> list[Path]:
    """Return any required inference artifacts that are missing from the repo."""
    return [path for path in get_required_artifacts(model_path) if not path.exists()]


def main() -> int:
    missing = find_missing_artifacts()
    if missing:
        print("Missing runtime artifacts:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("Runtime artifacts verified:")
    for path in get_required_artifacts():
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
