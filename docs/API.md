# API Guide

## Scope

This project now includes a FastAPI wrapper around `src/predict_risk.py` for
programmatic scoring.

The API is intended for:

- local engineering demos
- integration experiments
- portfolio walkthroughs

It is not intended for:

- clinical use
- diagnosis
- crisis triage

## Run The API Locally

Install the full project dependencies first:

```powershell
py -3 -m pip --python .\.venv\Scripts\python.exe install -r requirements.txt
```

Set an API key for protected access:

```powershell
$env:OHRE_API_KEY = "replace-with-a-strong-local-key"
```

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

Open the generated docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`
- Health endpoint: `http://127.0.0.1:8000/health`

## Environment Variables

- `OHRE_API_KEY`: required for protected `/predict` access
- `OHRE_RATE_LIMIT_PER_MINUTE`: per-client request cap, default `60`
- `OHRE_REQUEST_LOG_PATH`: optional file path for request logs

If `OHRE_API_KEY` is not set, the API still starts for local development, but
authentication is effectively disabled. Set it before sharing the service.

## Request Schema

`POST /predict`

```json
{
  "age": 35,
  "sex_female": 0,
  "poverty_ratio": 2.5,
  "met_min_week": 300,
  "sleep_hours": 7.0,
  "sleep_trouble": 0,
  "bmi": 24.0,
  "drinks_per_week": 3,
  "education": 4,
  "race_eth": 3
}
```

Key validation rules:

- `age`: 18 to 80
- `sex_female`: 0 or 1
- `poverty_ratio`: 0.0 to 5.0
- `met_min_week`: 0 to 3000
- `sleep_hours`: 3.0 to 12.0
- `sleep_trouble`: 0 or 1
- `bmi`: 15.0 to 50.0
- `drinks_per_week`: 0 to 40
- `education`: 1 to 5
- `race_eth`: one of `1, 2, 3, 4, 6, 7`

## Example Curl Request

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -H "X-API-Key: replace-with-a-strong-local-key" `
  -d "{\"age\":35,\"sex_female\":0,\"poverty_ratio\":2.5,\"met_min_week\":300,\"sleep_hours\":7.0,\"sleep_trouble\":0,\"bmi\":24.0,\"drinks_per_week\":3,\"education\":4,\"race_eth\":3}"
```

## Example Response

```json
{
  "risk_score": 0.3471,
  "risk_label": "Low risk",
  "risk_color": "blue",
  "phq9_estimate": 9.4,
  "top_factors": [
    {
      "feature": "sleep_trouble",
      "importance": 0.1042,
      "value": 0.0
    }
  ]
}
```

## Postman And Integration Stub

- Postman collection: `docs/OpenHealthRiskEngine.postman_collection.json`
- Browser/web-app fetch example: `docs/web_integration_stub.js`

## Operational Notes

- The API reuses the repo's trained model artifacts from `models/`
- Docker and Render builds now run `src/verify_runtime.py` to fail fast if
  required artifacts are missing
- Request logging is enabled through FastAPI middleware and writes to stdout by
  default
