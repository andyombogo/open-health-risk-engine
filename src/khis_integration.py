"""
khis_integration.py
-------------------
Bridge Kenya county KHIS signals into the Open Health Risk Engine baseline.

This module deliberately treats KHIS data as an aggregate county-level signal,
not a substitute for person-level clinical labels. The current OHRE model was
trained on NHANES individual records from the United States, so this bridge
creates a small synthetic county cohort from KHIS MNS activity and uses the
existing predictor only as a research baseline for comparative county scoring.
US-specific variables that do not translate cleanly to Kenya are pinned to
neutral reference values to avoid injecting arbitrary bias into the baseline.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.predict_risk import RiskPredictor

MNS_SEARCH_TERMS = (
    "mental",
    "depression",
    "psychiatry",
    "psychosis",
    "epilepsy",
    "substance",
    "alcohol",
    "drug",
    "suicide",
)
FALLBACK_MNS_INDICATORS = (
    {
        "id": "khis_mns_contacts",
        "name": "MNS outpatient contacts",
        "short_name": "MNS contacts",
        "code": "KHIS_MNS_CONTACTS",
        "description": "Fallback proxy indicator for mental health service contacts.",
    },
    {
        "id": "khis_substance_use_visits",
        "name": "Substance use follow-up visits",
        "short_name": "Substance follow-up",
        "code": "KHIS_SUBSTANCE_VISITS",
        "description": "Fallback proxy indicator for substance-use workload.",
    },
    {
        "id": "khis_neuro_psych_reviews",
        "name": "Neurology and psychiatry review visits",
        "short_name": "Neuro-psych reviews",
        "code": "KHIS_NEURO_PSYCH",
        "description": "Fallback proxy indicator for neuropsychiatric review burden.",
    },
)
PROFILE_TEMPLATES = (
    {
        "profile_name": "working_age_adult",
        "age": 34.0,
        "sex_female": 0,
        "poverty_ratio": 2.3,
        "marital_status": 1,
        "born_us": 1,
        "met_min_week": 360.0,
        "sedentary_minutes": 360.0,
        "sleep_hours": 7.1,
        "sleep_trouble": 0,
        "sleep_apnea_symptom_freq": 0.0,
        "daytime_sleepiness_freq": 1.0,
        "bmi": 25.5,
        "drinks_per_week": 2.0,
        "ever_smoked_100_cigs": 0,
        "current_smoking_status": 3.0,
        "days_smoked_past_month": 0.0,
        "cigs_per_day_smoking_days": 0.0,
        "quit_attempt_last_year": 0,
        "general_health": 2.0,
        "healthcare_visits_code": 2.0,
        "hospitalized_last_year": 0,
        "routine_care_place": 1.0,
        "insurance_gap_last_year": 0,
        "insured": 1,
        "education": 3.0,
        "race_eth": 3,
    },
    {
        "profile_name": "working_age_female",
        "age": 37.0,
        "sex_female": 1,
        "poverty_ratio": 2.0,
        "marital_status": 1,
        "born_us": 1,
        "met_min_week": 300.0,
        "sedentary_minutes": 390.0,
        "sleep_hours": 6.9,
        "sleep_trouble": 0,
        "sleep_apnea_symptom_freq": 0.0,
        "daytime_sleepiness_freq": 1.0,
        "bmi": 27.0,
        "drinks_per_week": 1.0,
        "ever_smoked_100_cigs": 0,
        "current_smoking_status": 3.0,
        "days_smoked_past_month": 0.0,
        "cigs_per_day_smoking_days": 0.0,
        "quit_attempt_last_year": 0,
        "general_health": 2.0,
        "healthcare_visits_code": 2.0,
        "hospitalized_last_year": 0,
        "routine_care_place": 1.0,
        "insurance_gap_last_year": 0,
        "insured": 1,
        "education": 3.0,
        "race_eth": 3,
    },
    {
        "profile_name": "older_vulnerable",
        "age": 61.0,
        "sex_female": 1,
        "poverty_ratio": 1.4,
        "marital_status": 2,
        "born_us": 1,
        "met_min_week": 120.0,
        "sedentary_minutes": 480.0,
        "sleep_hours": 6.3,
        "sleep_trouble": 1,
        "sleep_apnea_symptom_freq": 1.0,
        "daytime_sleepiness_freq": 2.0,
        "bmi": 29.0,
        "drinks_per_week": 0.0,
        "ever_smoked_100_cigs": 0,
        "current_smoking_status": 3.0,
        "days_smoked_past_month": 0.0,
        "cigs_per_day_smoking_days": 0.0,
        "quit_attempt_last_year": 0,
        "general_health": 3.0,
        "healthcare_visits_code": 3.0,
        "hospitalized_last_year": 0,
        "routine_care_place": 1.0,
        "insurance_gap_last_year": 0,
        "insured": 1,
        "education": 2.0,
        "race_eth": 3,
    },
    {
        "profile_name": "care_gap_stress_case",
        "age": 43.0,
        "sex_female": 1,
        "poverty_ratio": 1.0,
        "marital_status": 5,
        "born_us": 1,
        "met_min_week": 60.0,
        "sedentary_minutes": 540.0,
        "sleep_hours": 5.8,
        "sleep_trouble": 1,
        "sleep_apnea_symptom_freq": 1.0,
        "daytime_sleepiness_freq": 2.0,
        "bmi": 30.5,
        "drinks_per_week": 4.0,
        "ever_smoked_100_cigs": 1,
        "current_smoking_status": 2.0,
        "days_smoked_past_month": 8.0,
        "cigs_per_day_smoking_days": 4.0,
        "quit_attempt_last_year": 0,
        "general_health": 3.0,
        "healthcare_visits_code": 2.0,
        "hospitalized_last_year": 0,
        "routine_care_place": 2.0,
        "insurance_gap_last_year": 1,
        "insured": 0,
        "education": 2.0,
        "race_eth": 3,
    },
)


def load_khis_mental_health(
    connector: Any, counties: Iterable[str] | str
) -> pd.DataFrame:
    """Load KHIS MNS activity and convert it into a batch-scoring baseline cohort.

    Parameters
    ----------
    connector:
        A connected `khis.DHIS2Connector` instance.
    counties:
        County names to include. A string, comma-separated string, or iterable
        of county names is accepted.

    Returns
    -------
    pandas.DataFrame
        A synthetic county proxy cohort with one row per county-profile pair,
        ready for `RiskPredictor.predict_batch()`.
    """
    khis = _import_khis()
    county_names = _normalise_counties(khis, counties)
    indicators = _discover_mns_indicators(connector)
    analytics = _fetch_county_mns_analytics(
        connector=connector,
        khis=khis,
        counties=county_names,
        indicators=indicators,
    )
    cleaned = khis.clean(analytics)
    summary = _summarise_county_signals(khis, cleaned, county_names)
    return _build_proxy_profiles(summary)


def score_county_risk(counties_df: pd.DataFrame) -> pd.DataFrame:
    """Score the proxy cohort and aggregate results back to the county level."""
    if counties_df.empty:
        return pd.DataFrame(
            columns=[
                "county",
                "county_code",
                "region",
                "org_unit_id",
                "mean_risk_score",
                "pct_high_risk",
                "mean_phq9_estimate",
                "max_risk_score",
                "risk_label",
                "risk_color",
                "n_proxy_profiles",
                "mns_burden_index",
                "service_pressure_index",
                "mns_total_contacts",
                "mns_indicator_count",
            ]
        )

    predictor = RiskPredictor()
    scored = predictor.predict_batch(counties_df)
    summary = (
        scored.groupby(
            ["county", "county_code", "region", "org_unit_id"], as_index=False
        )
        .agg(
            mean_risk_score=("risk_score", "mean"),
            pct_high_risk=(
                "above_decision_threshold",
                lambda series: float(pd.Series(series).astype(float).mean() * 100.0),
            ),
            mean_phq9_estimate=("phq9_estimate", "mean"),
            max_risk_score=("risk_score", "max"),
            n_proxy_profiles=("profile_name", "count"),
            mns_burden_index=("mns_burden_index", "mean"),
            service_pressure_index=("service_pressure_index", "mean"),
            mns_total_contacts=("mns_total_contacts", "mean"),
            mns_indicator_count=("mns_indicator_count", "mean"),
        )
        .sort_values(["mean_risk_score", "pct_high_risk"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["mean_risk_score"] = summary["mean_risk_score"].round(4)
    summary["pct_high_risk"] = summary["pct_high_risk"].round(1)
    summary["mean_phq9_estimate"] = summary["mean_phq9_estimate"].round(2)
    summary["max_risk_score"] = summary["max_risk_score"].round(4)
    summary["mns_burden_index"] = summary["mns_burden_index"].round(4)
    summary["service_pressure_index"] = summary["service_pressure_index"].round(4)
    summary["mns_total_contacts"] = summary["mns_total_contacts"].round(2)
    summary["mns_indicator_count"] = summary["mns_indicator_count"].round(0).astype(int)

    labels = summary["mean_risk_score"].apply(predictor.get_severity_label)
    summary["risk_label"] = labels.str[0]
    summary["risk_color"] = labels.str[1]
    return summary


def _import_khis():
    """Import `khis`, using the sibling repo if it is not installed globally."""
    try:
        import khis  # type: ignore

        return khis
    except ImportError:
        sibling_repo = Path(__file__).resolve().parents[2] / "khis-toolkit"
        if sibling_repo.exists():
            sys.path.insert(0, str(sibling_repo))
            import khis  # type: ignore

            return khis
        raise


def _normalise_counties(khis, counties: Iterable[str] | str) -> list[str]:
    """Resolve county inputs into canonical Kenya county names."""
    if isinstance(counties, str):
        raw_values = [part.strip() for part in counties.split(",") if part.strip()]
    else:
        raw_values = [str(part).strip() for part in counties if str(part).strip()]

    if not raw_values:
        raise ValueError("Pass at least one county name.")

    canonical: list[str] = []
    for name in raw_values:
        county_record = khis.get_county(name)
        canonical.append(str(county_record["name"]))
    return canonical


def _discover_mns_indicators(connector: Any) -> pd.DataFrame:
    """Discover KHIS MNS indicators, or return a small fallback catalog."""
    indicator_frames: list[pd.DataFrame] = []
    for term in MNS_SEARCH_TERMS:
        try:
            matches = connector.get_indicators(search_term=term)
        except Exception:
            continue
        if matches.empty:
            continue
        indicator_frames.append(matches.assign(search_term=term))

    if not indicator_frames:
        return pd.DataFrame.from_records(FALLBACK_MNS_INDICATORS)

    combined = (
        pd.concat(indicator_frames, ignore_index=True)
        .drop_duplicates(subset=["id"])
        .reset_index(drop=True)
    )
    return combined.head(6)


def _fetch_county_mns_analytics(
    connector: Any,
    khis,
    counties: list[str],
    indicators: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch live KHIS analytics when possible, otherwise build a stable fallback."""
    if getattr(connector, "using_demo_server", False):
        return _build_fallback_analytics(khis, counties, indicators)

    resolved_pairs: list[tuple[str, str]] = []
    for county in counties:
        try:
            org_unit_id = connector.resolve_org_unit_id_by_name(county)
        except Exception:
            try:
                org_unit_id = khis.resolve_org_unit_id(county)
            except Exception:
                continue
        resolved_pairs.append((county, org_unit_id))

    if not resolved_pairs:
        return _build_fallback_analytics(khis, counties, indicators)

    org_unit_lookup = {org_unit_id: county for county, org_unit_id in resolved_pairs}
    indicator_ids = indicators["id"].astype(str).tolist()
    try:
        analytics = connector.get_analytics(
            indicator_ids=indicator_ids,
            org_unit_ids=list(org_unit_lookup.keys()),
            periods="LAST_12_MONTHS",
        )
    except Exception:
        analytics = pd.DataFrame()

    if analytics.empty:
        return _build_fallback_analytics(khis, counties, indicators)

    analytics = analytics.copy()
    analytics["org_unit_name"] = (
        analytics["org_unit_id"]
        .astype(str)
        .map(org_unit_lookup)
        .fillna(analytics["org_unit_name"])
    )
    return analytics


def _build_fallback_analytics(
    khis,
    counties: list[str],
    indicators: pd.DataFrame,
) -> pd.DataFrame:
    """Create deterministic county MNS proxy analytics for demo and offline use."""
    county_meta = khis.list_counties().rename(columns={"name": "county"})
    selected_meta = county_meta[county_meta["county"].isin(counties)].copy()
    if selected_meta.empty:
        selected_meta = county_meta.head(len(counties)).copy()
        selected_meta["county"] = counties[: len(selected_meta)]

    periods = pd.date_range("2024-01-01", periods=12, freq="MS")
    selected_indicators = indicators.head(3).reset_index(drop=True)
    if selected_indicators.empty:
        selected_indicators = pd.DataFrame.from_records(FALLBACK_MNS_INDICATORS)

    records: list[dict[str, object]] = []
    for county_index, county_row in enumerate(selected_meta.to_dict(orient="records")):
        county_code = int(county_row["code"])
        county_multiplier = 0.92 + (county_code % 9) * 0.08
        for indicator_index, indicator_row in selected_indicators.iterrows():
            indicator_multiplier = 1.0 + indicator_index * 0.35
            for month_index, period in enumerate(periods):
                seasonal = ((period.month + county_index) % 6) * 1.25
                trend = month_index * (0.35 + indicator_index * 0.05)
                value = round(
                    (14.0 * county_multiplier * indicator_multiplier)
                    + seasonal
                    + trend,
                    2,
                )
                records.append(
                    {
                        "indicator_id": str(indicator_row["id"]),
                        "indicator_name": str(indicator_row["name"]),
                        "org_unit_id": str(
                            khis.resolve_org_unit_id(county_row["county"])
                        ),
                        "org_unit_name": str(county_row["county"]),
                        "period": period.strftime("%Y%m"),
                        "value": value,
                    }
                )

    return pd.DataFrame.from_records(records)


def _summarise_county_signals(
    khis,
    cleaned: pd.DataFrame,
    counties: list[str],
) -> pd.DataFrame:
    """Aggregate cleaned county MNS activity into stable county burden features."""
    county_meta = khis.list_counties().rename(columns={"name": "county"})
    cleaned = cleaned.copy()
    cleaned["county"] = cleaned["org_unit_name"].astype(str)

    indicator_counts = (
        cleaned.groupby("county")["indicator_name"]
        .nunique()
        .rename("mns_indicator_count")
    )
    monthly = (
        cleaned.groupby(["county", "period"], as_index=False)["value"]
        .sum()
        .sort_values(["county", "period"], kind="mergesort")
    )
    first_last = (
        monthly.groupby("county")
        .agg(
            first_month_value=("value", "first"),
            last_month_value=("value", "last"),
            mns_total_contacts=("value", "sum"),
            mean_monthly_contacts=("value", "mean"),
            std_monthly_contacts=("value", "std"),
            periods_reported=("period", "nunique"),
        )
        .reset_index()
    )
    first_last["std_monthly_contacts"] = first_last["std_monthly_contacts"].fillna(0.0)
    first_last["trend_ratio"] = (
        first_last["last_month_value"] - first_last["first_month_value"]
    ) / first_last["mean_monthly_contacts"].replace(0, np.nan)
    first_last["trend_ratio"] = (
        first_last["trend_ratio"].replace([np.inf, -np.inf], 0).fillna(0.0)
    )
    first_last["volatility_index"] = first_last["std_monthly_contacts"] / first_last[
        "mean_monthly_contacts"
    ].replace(0, np.nan)
    first_last["volatility_index"] = (
        first_last["volatility_index"].replace([np.inf, -np.inf], 0).fillna(0.0)
    )
    summary = first_last.merge(indicator_counts, on="county", how="left")
    summary["mns_indicator_count"] = (
        summary["mns_indicator_count"].fillna(1).astype(int)
    )

    summary["burden_raw"] = (
        np.log1p(summary["mns_total_contacts"])
        + 0.35 * summary["mns_indicator_count"]
        + 0.08 * summary["periods_reported"]
        + 0.9 * summary["volatility_index"]
        + 0.7 * summary["trend_ratio"].clip(lower=0)
    )
    summary["mns_burden_index"] = _min_max_scale(summary["burden_raw"])
    summary["service_pressure_index"] = _min_max_scale(
        summary["mean_monthly_contacts"] + summary["last_month_value"]
    )

    summary = county_meta.merge(summary, on="county", how="right")
    summary["org_unit_id"] = summary["county"].apply(khis.resolve_org_unit_id)

    if len(summary) < len(counties):
        missing = [
            county for county in counties if county not in summary["county"].tolist()
        ]
        if missing:
            fallback = county_meta[county_meta["county"].isin(missing)].copy()
            fallback["first_month_value"] = 0.0
            fallback["last_month_value"] = 0.0
            fallback["mns_total_contacts"] = 0.0
            fallback["mean_monthly_contacts"] = 0.0
            fallback["std_monthly_contacts"] = 0.0
            fallback["periods_reported"] = 0
            fallback["trend_ratio"] = 0.0
            fallback["volatility_index"] = 0.0
            fallback["mns_indicator_count"] = 1
            fallback["burden_raw"] = 0.0
            fallback["mns_burden_index"] = 0.0
            fallback["service_pressure_index"] = 0.0
            fallback["org_unit_id"] = fallback["county"].apply(khis.resolve_org_unit_id)
            summary = pd.concat([summary, fallback], ignore_index=True)

    summary = summary[summary["county"].isin(counties)].copy()
    summary["county"] = pd.Categorical(
        summary["county"], categories=counties, ordered=True
    )
    return summary.sort_values("county", kind="mergesort").reset_index(drop=True)


def _build_proxy_profiles(summary: pd.DataFrame) -> pd.DataFrame:
    """Expand county burden summaries into a small synthetic cohort per county."""
    records: list[dict[str, object]] = []
    for county_row in summary.to_dict(orient="records"):
        burden = float(county_row["mns_burden_index"])
        service_pressure = float(county_row["service_pressure_index"])
        volatility = float(county_row["volatility_index"])
        trend_ratio = float(county_row["trend_ratio"])

        for template in PROFILE_TEMPLATES:
            row = dict(template)
            row["county"] = str(county_row["county"])
            row["county_code"] = int(county_row["code"])
            row["region"] = str(county_row["region"])
            row["county_capital"] = str(county_row["capital"])
            row["org_unit_id"] = str(county_row["org_unit_id"])
            row["mns_burden_index"] = round(burden, 4)
            row["service_pressure_index"] = round(service_pressure, 4)
            row["volatility_index"] = round(volatility, 4)
            row["trend_ratio"] = round(trend_ratio, 4)
            row["mns_total_contacts"] = float(county_row["mns_total_contacts"])
            row["mns_indicator_count"] = int(county_row["mns_indicator_count"])
            row["periods_reported"] = int(county_row["periods_reported"])
            row["age"] = _clamp(row["age"] + burden * 2.0, 18.0, 80.0)
            row["poverty_ratio"] = _clamp(row["poverty_ratio"] - burden * 1.1, 0.2, 5.0)
            row["met_min_week"] = _clamp(
                row["met_min_week"] * (1.0 - 0.55 * burden),
                0.0,
                1200.0,
            )
            row["sedentary_minutes"] = _clamp(
                row["sedentary_minutes"] + 170.0 * burden + 60.0 * volatility,
                120.0,
                900.0,
            )
            row["sleep_hours"] = _clamp(
                row["sleep_hours"] - (1.0 * burden) - max(trend_ratio, 0.0) * 0.4,
                4.5,
                9.5,
            )
            row["sleep_trouble"] = int(
                row["sleep_trouble"] or burden >= 0.45 or volatility >= 0.55
            )
            row["sleep_apnea_symptom_freq"] = _clamp(
                round(row["sleep_apnea_symptom_freq"] + (2.0 * burden) + volatility),
                0,
                4,
            )
            row["daytime_sleepiness_freq"] = _clamp(
                round(
                    row["daytime_sleepiness_freq"]
                    + (2.0 * burden)
                    + max(trend_ratio, 0.0)
                ),
                0,
                4,
            )
            row["bmi"] = _clamp(
                row["bmi"] + (4.0 * burden) + (1.5 * volatility),
                18.0,
                42.0,
            )
            row["drinks_per_week"] = _clamp(
                row["drinks_per_week"] + (4.0 * max(trend_ratio, 0.0)) + (2.0 * burden),
                0.0,
                28.0,
            )
            row["ever_smoked_100_cigs"] = int(
                row["ever_smoked_100_cigs"] or burden >= 0.6
            )
            if burden >= 0.7 and row["current_smoking_status"] == 3.0:
                row["current_smoking_status"] = 2.0
            row["days_smoked_past_month"] = _clamp(
                row["days_smoked_past_month"]
                + (
                    10.0 * burden
                    if row["current_smoking_status"] in {1.0, 2.0}
                    else 0.0
                ),
                0.0,
                30.0,
            )
            row["cigs_per_day_smoking_days"] = _clamp(
                row["cigs_per_day_smoking_days"]
                + (
                    5.0 * burden if row["current_smoking_status"] in {1.0, 2.0} else 0.0
                ),
                0.0,
                20.0,
            )
            row["quit_attempt_last_year"] = int(
                row["quit_attempt_last_year"]
                or (burden >= 0.55 and row["current_smoking_status"] in {1.0, 2.0})
            )
            row["general_health"] = _clamp(
                round(row["general_health"] + (2.0 * burden) + volatility),
                1,
                5,
            )
            row["healthcare_visits_code"] = _clamp(
                round(row["healthcare_visits_code"] + (2.0 * service_pressure)),
                0,
                5,
            )
            row["hospitalized_last_year"] = int(
                row["hospitalized_last_year"] or service_pressure >= 0.8
            )
            row["routine_care_place"] = (
                2.0 if (row["routine_care_place"] != 1.0 or burden >= 0.75) else 1.0
            )
            row["insurance_gap_last_year"] = int(
                row["insurance_gap_last_year"] or burden >= 0.65
            )
            row["insured"] = int(not row["insurance_gap_last_year"] or burden < 0.8)
            row["education"] = _clamp(round(row["education"] - burden), 1, 5)
            records.append(row)

    return pd.DataFrame.from_records(records)


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0-1, using 0.5 for constant inputs."""
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - minimum) / (maximum - minimum)


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to the provided inclusive interval."""
    return float(max(lower, min(upper, value)))
