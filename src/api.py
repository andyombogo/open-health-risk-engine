"""FastAPI wrapper for scoring the deployed risk model."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from src.predict_risk import (
    RiskPredictor,
    resolve_decision_threshold,
    resolve_model_path,
)
from src.verify_runtime import find_missing_artifacts

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / ".tmp"
DEFAULT_RATE_LIMIT_PER_MINUTE = 60


class PredictRequest(BaseModel):
    """Validated schema for a single risk-scoring request."""

    age: int = Field(..., ge=18, le=80, description="Age in years.")
    sex_female: int = Field(
        ...,
        ge=0,
        le=1,
        description="Use 1 for female, 0 for male to match the trained features.",
    )
    poverty_ratio: float = Field(
        ...,
        ge=0.0,
        le=5.0,
        description="Household poverty-income ratio.",
    )
    met_min_week: float = Field(
        ...,
        ge=0.0,
        le=3000.0,
        description="Weekly physical activity in MET-minutes.",
    )
    sleep_hours: float = Field(
        ...,
        ge=3.0,
        le=12.0,
        description="Average sleep hours per night.",
    )
    sleep_trouble: int = Field(
        ...,
        ge=0,
        le=1,
        description="Use 1 if the respondent regularly has trouble sleeping.",
    )
    bmi: float = Field(..., ge=15.0, le=50.0, description="Body mass index.")
    drinks_per_week: float = Field(
        ...,
        ge=0.0,
        le=40.0,
        description="Estimated drinks consumed per week.",
    )
    education: int = Field(
        ...,
        ge=1,
        le=5,
        description="Ordinal NHANES education bucket from 1 to 5.",
    )
    race_eth: int = Field(
        ...,
        description="NHANES race/ethnicity code: 1, 2, 3, 4, 6, or 7.",
    )

    @field_validator("race_eth")
    @classmethod
    def validate_race_eth(cls, value: int) -> int:
        allowed = {1, 2, 3, 4, 6, 7}
        if value not in allowed:
            raise ValueError(f"race_eth must be one of {sorted(allowed)}")
        return value


class RiskFactorResponse(BaseModel):
    """Single feature-importance item in the prediction response."""

    feature: str
    importance: float
    value: float


class PredictResponse(BaseModel):
    """Structured API response for one prediction."""

    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_label: str
    risk_color: str
    phq9_estimate: float = Field(..., ge=0.0, le=27.0)
    decision_threshold: float = Field(..., ge=0.0, le=1.0)
    above_decision_threshold: bool
    top_factors: list[RiskFactorResponse]


class HealthResponse(BaseModel):
    """Health and configuration details for deployment checks."""

    status: str
    model_loaded: bool
    missing_artifacts: list[str]
    auth_enabled: bool
    rate_limit_per_minute: int
    model_artifact: str
    decision_threshold: float = Field(..., ge=0.0, le=1.0)
    calibrated_scores: bool


@dataclass(frozen=True)
class APISettings:
    """Runtime configuration for the API surface."""

    api_key: str | None
    rate_limit_per_minute: int
    request_log_path: Path | None
    model_path: Path
    decision_threshold: float


class InMemoryRateLimiter:
    """Simple in-process rate limiter keyed by client identity."""

    def __init__(self, limit_per_minute: int):
        self.limit_per_minute = max(1, limit_per_minute)
        self._history: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        window_start = now - 60
        with self._lock:
            history = self._history[key]
            while history and history[0] < window_start:
                history.popleft()
            if len(history) >= self.limit_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        "Rate limit exceeded. Reduce request frequency or raise "
                        "OHRE_RATE_LIMIT_PER_MINUTE for trusted callers."
                    ),
                )
            history.append(now)


def build_settings(
    api_key: str | None = None,
    rate_limit_per_minute: int | None = None,
    request_log_path: str | Path | None = None,
    model_path: str | Path | None = None,
    decision_threshold: float | None = None,
) -> APISettings:
    """Build runtime settings from overrides or environment variables."""

    if rate_limit_per_minute is None:
        rate_limit_per_minute = int(
            os.getenv("OHRE_RATE_LIMIT_PER_MINUTE", str(DEFAULT_RATE_LIMIT_PER_MINUTE))
        )
    if api_key is None:
        api_key = os.getenv("OHRE_API_KEY")
    if request_log_path is None:
        env_log_path = os.getenv("OHRE_REQUEST_LOG_PATH", "").strip()
        request_log_path = Path(env_log_path) if env_log_path else None
    resolved_model_path = resolve_model_path(model_path)
    resolved_decision_threshold = resolve_decision_threshold(decision_threshold)

    return APISettings(
        api_key=api_key.strip() if api_key else None,
        rate_limit_per_minute=rate_limit_per_minute,
        request_log_path=Path(request_log_path) if request_log_path else None,
        model_path=resolved_model_path,
        decision_threshold=resolved_decision_threshold,
    )


def configure_logger(log_path: Path | None = None) -> logging.Logger:
    """Create a request logger that writes to stdout and optionally a file."""

    logger = logging.getLogger("open_health_risk_engine.api")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@lru_cache(maxsize=8)
def get_predictor(
    model_path: str | Path | None = None,
    decision_threshold: float | None = None,
) -> RiskPredictor:
    """Load the trained predictor or raise a service-level error."""

    resolved_model_path = resolve_model_path(model_path)
    resolved_decision_threshold = resolve_decision_threshold(decision_threshold)
    missing = find_missing_artifacts(resolved_model_path)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Required model artifacts are missing. Run the training pipeline "
                "or bundle the models directory for deployment."
            ),
        )
    try:
        return RiskPredictor(
            model_path=resolved_model_path,
            decision_threshold=resolved_decision_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def create_app(
    api_key: str | None = None,
    rate_limit_per_minute: int | None = None,
    request_log_path: str | Path | None = None,
    model_path: str | Path | None = None,
    decision_threshold: float | None = None,
) -> FastAPI:
    """Create a configured FastAPI application instance."""

    settings = build_settings(
        api_key=api_key,
        rate_limit_per_minute=rate_limit_per_minute,
        request_log_path=request_log_path,
        model_path=model_path,
        decision_threshold=decision_threshold,
    )
    logger = configure_logger(settings.request_log_path)
    limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)

    app = FastAPI(
        title="Open Health Risk Engine API",
        version="0.1.0",
        description=(
            "Programmatic scoring interface for the Open Health Risk Engine "
            "portfolio demo. This API is for research and engineering demos only."
        ),
        contact={
            "name": "Open Health Risk Engine",
            "url": "https://github.com/andyombogo/open-health-risk-engine",
        },
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            "%s %s status=%s client=%s duration_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            client_host,
            duration_ms,
        )
        return response

    def authorize_and_limit(
        request: Request,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid X-API-Key header.",
            )
        client_host = request.client.host if request.client else "unknown"
        rate_key = x_api_key or client_host
        limiter.check(rate_key)

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str | bool]:
        return {
            "project": "Open Health Risk Engine",
            "docs_url": "/docs",
            "openapi_url": "/openapi.json",
            "auth_enabled": bool(settings.api_key),
        }

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        missing = []
        for path in find_missing_artifacts(settings.model_path):
            try:
                missing.append(str(path.relative_to(ROOT)))
            except ValueError:
                missing.append(str(path))
        return HealthResponse(
            status="ok" if not missing else "degraded",
            model_loaded=not missing,
            missing_artifacts=missing,
            auth_enabled=bool(settings.api_key),
            rate_limit_per_minute=settings.rate_limit_per_minute,
            model_artifact=settings.model_path.name,
            decision_threshold=settings.decision_threshold,
            calibrated_scores="calibrated" in settings.model_path.name,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["inference"],
        dependencies=[Depends(authorize_and_limit)],
    )
    def predict(payload: PredictRequest) -> PredictResponse:
        predictor = get_predictor(settings.model_path, settings.decision_threshold)
        result = predictor.predict(payload.model_dump())
        return PredictResponse(**result)

    return app


app = create_app()


__all__ = [
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "RiskFactorResponse",
    "app",
    "create_app",
]
