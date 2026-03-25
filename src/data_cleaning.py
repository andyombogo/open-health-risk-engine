"""
data_cleaning.py
----------------
Loads, merges, and cleans the NHANES XPT files into a single analysis-ready DataFrame.

Epidemiological decisions made here (all documented):
  - Exclusion criteria: age < 18, pregnancy, missing PHQ-9
  - PHQ-9 coding: NHANES codes 7 (refused) and 9 (don't know) set to NaN
  - Income: use poverty-income ratio (PIR) as continuous variable
  - Physical activity: compute total MET-minutes/week from vigorous + moderate + walk
  - Add smoking, general-health, and insurance variables because they add
    stronger non-lifestyle signal than threshold tuning alone
  - Missing data strategy: mean imputation for continuous, mode for categorical
    (a more sophisticated study would use multiple imputation — see ROADMAP.md)

Output: data/processed/nhanes_clean.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# NHANES missing/refused codes to treat as NaN
NHANES_MISSING_CODES = [7777, 9999, 77777, 99999, 7, 9]


def load_xpt(filename: str) -> pd.DataFrame:
    """Load a NHANES XPT file."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_data.py first."
        )
    return pd.read_sas(path, format="xport", encoding="utf-8")


def clean_phq9(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the PHQ-9 depression screener file (DPQ_J).

    PHQ-9 items: DPQ010 to DPQ090
    Each item scored 0-3. Sum = total PHQ-9 score (0-27).
    Standard severity categories:
      0-4   = minimal/none
      5-9   = mild
      10-14 = moderate
      15-19 = moderately severe
      20-27 = severe
    """
    phq_cols = [f"DPQ0{i}0" for i in range(1, 10)]

    # Replace NHANES refused/unknown codes with NaN
    df[phq_cols] = df[phq_cols].replace(NHANES_MISSING_CODES, np.nan)

    # Compute total PHQ-9 score — require at least 8 of 9 items answered
    df["phq9_score"] = df[phq_cols].apply(
        lambda row: row.sum() if row.notna().sum() >= 8 else np.nan, axis=1
    )

    # Severity category (clinically standard thresholds)
    df["phq9_severity"] = pd.cut(
        df["phq9_score"],
        bins=[-1, 4, 9, 14, 19, 27],
        labels=["minimal", "mild", "moderate", "moderately_severe", "severe"],
    )

    # Binary outcome: moderate or worse (PHQ-9 >= 10) — used for classification
    df["depression_binary"] = (df["phq9_score"] >= 10).astype(int)

    return df[["SEQN", "phq9_score", "phq9_severity", "depression_binary"]]


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DEMO_J demographics file.
    Key variables: age, sex, race/ethnicity, education, poverty-income ratio.
    """
    df = df.copy()

    # Age: RIDAGEYR is exact age in years — exclude under 18
    df = df[df["RIDAGEYR"] >= 18].copy()

    df = df.replace(NHANES_MISSING_CODES, np.nan)

    # Rename for clarity
    df = df.rename(columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "sex",          # 1=Male, 2=Female
        "RIDRETH3": "race_eth",     # 1=Mexican American, 2=Other Hispanic,
                                    # 3=Non-Hispanic White, 4=Non-Hispanic Black,
                                    # 6=Non-Hispanic Asian, 7=Other/Multi
        "DMDEDUC2": "education",    # 1=<9th grade ... 5=college+
        "INDFMPIR": "poverty_ratio", # Poverty income ratio (continuous)
        "DMDMARTZ": "marital_status",
        "DMDBORN4": "born_us",
    })

    # Exclude pregnant participants (RIDEXPRG == 1)
    if "RIDEXPRG" in df.columns:
        df = df[df["RIDEXPRG"] != 1].copy()

    # Sex: encode as binary (0=Male, 1=Female)
    df["sex_female"] = (df["sex"] == 2).astype(int)

    # Education: treat 7/9 (refused/don't know) as missing
    df["education"] = df["education"].replace([7, 9], np.nan)

    # Poverty ratio: cap at 5.0 (top-coded by NHANES), floor at 0
    df["poverty_ratio"] = df["poverty_ratio"].clip(0, 5.0)

    return df[
        [
            "SEQN",
            "age",
            "sex_female",
            "race_eth",
            "education",
            "poverty_ratio",
            "marital_status",
            "born_us",
        ]
    ]


def clean_physical_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean PAQ_J physical activity file.
    Compute total MET-minutes/week (standard epidemiological measure).

    MET values (standard from WHO / Ainsworth compendium):
      Vigorous activity: 8 MET
      Moderate activity: 4 MET
      Walking:           3.5 MET
    """
    df = df.copy()

    # Replace missing codes
    for col in df.columns:
        df[col] = df[col].replace(NHANES_MISSING_CODES, np.nan)

    # Vigorous recreational activity (days/week × minutes/day)
    vig_days = df.get("PAQ650", pd.Series(np.nan, index=df.index))
    vig_min = df.get("PAQ655", pd.Series(np.nan, index=df.index))
    vig_met = vig_days.fillna(0) * vig_min.fillna(0) * 8

    # Moderate recreational activity
    mod_days = df.get("PAQ665", pd.Series(np.nan, index=df.index))
    mod_min = df.get("PAQ670", pd.Series(np.nan, index=df.index))
    mod_met = mod_days.fillna(0) * mod_min.fillna(0) * 4

    # Walking
    walk_days = df.get("PAD660", pd.Series(np.nan, index=df.index))
    walk_min = df.get("PAD675", pd.Series(np.nan, index=df.index))
    walk_met = walk_days.fillna(0) * walk_min.fillna(0) * 3.5

    df["met_min_week"] = vig_met + mod_met + walk_met

    # WHO physical activity categories
    df["activity_category"] = pd.cut(
        df["met_min_week"],
        bins=[-1, 0, 599, 1199, float("inf")],
        labels=["inactive", "insufficient", "sufficient", "highly_active"],
    )

    df["sedentary_minutes"] = pd.to_numeric(df.get("PAD680"), errors="coerce")

    return df[["SEQN", "met_min_week", "activity_category", "sedentary_minutes"]]


def clean_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean SLQ_J sleep file.
    SLD012: weekday sleep hours (self-reported)
    SLD013: weekend sleep hours
    """
    df = df.copy()
    df = df.replace(NHANES_MISSING_CODES, np.nan)

    # Average sleep (weighted: 5 weekdays, 2 weekend)
    weekday = df.get("SLD012", pd.Series(np.nan, index=df.index))
    weekend = df.get("SLD013", pd.Series(np.nan, index=df.index))
    df["sleep_hours_avg"] = (weekday * 5 + weekend * 2) / 7

    # Sleep trouble (binary): SLQ050 == 1 (yes)
    df["sleep_trouble"] = (df.get("SLQ050", pd.Series(2, index=df.index)) == 1).astype(int)
    df["sleep_apnea_symptom_freq"] = pd.to_numeric(df.get("SLQ040"), errors="coerce")
    df["daytime_sleepiness_freq"] = pd.to_numeric(df.get("SLQ120"), errors="coerce")

    return df[
        [
            "SEQN",
            "sleep_hours_avg",
            "sleep_trouble",
            "sleep_apnea_symptom_freq",
            "daytime_sleepiness_freq",
        ]
    ]


def clean_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean BMX_J body measures file.
    BMXBMI: Body mass index (kg/m²)
    """
    df = df.copy()
    df["bmi"] = pd.to_numeric(df.get("BMXBMI"), errors="coerce")

    # WHO BMI categories
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25.0, 30.0, float("inf")],
        labels=["underweight", "normal", "overweight", "obese"],
    )

    return df[["SEQN", "bmi", "bmi_category"]]


def clean_alcohol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ALQ_J alcohol use file.
    ALQ130: Average number of drinks per day on days drank
    ALQ120Q: How often drink over past 12 months
    """
    df = df.copy()
    df = df.replace(NHANES_MISSING_CODES, np.nan)

    df["drinks_per_day"] = pd.to_numeric(df.get("ALQ130"), errors="coerce")
    df["drink_frequency"] = pd.to_numeric(df.get("ALQ120Q"), errors="coerce")

    # Hazardous drinking flag (NIAAA definition: >14/week men, >7/week women)
    # We'll compute a simple weekly estimate; sex-specific threshold applied in feature_engineering
    df["drinks_per_week_est"] = df["drinks_per_day"].fillna(0) * df["drink_frequency"].fillna(0) / 52

    return df[["SEQN", "drinks_per_day", "drink_frequency", "drinks_per_week_est"]]


def clean_smoking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean SMQ_J smoking file.
    Includes ever-smoking, current smoking, smoking intensity, and quit attempts.
    """
    df = df.copy()
    df = df.replace([777, 999, *NHANES_MISSING_CODES], np.nan)

    df["ever_smoked_100_cigs"] = (df.get("SMQ020", pd.Series(2, index=df.index)) == 1).astype(int)
    df["current_smoking_status"] = pd.to_numeric(df.get("SMQ040"), errors="coerce")
    df["days_smoked_past_month"] = pd.to_numeric(df.get("SMD641"), errors="coerce")
    df["cigs_per_day_smoking_days"] = pd.to_numeric(df.get("SMD650"), errors="coerce")
    df["quit_attempt_last_year"] = (df.get("SMQ670", pd.Series(2, index=df.index)) == 1).astype(int)

    return df[
        [
            "SEQN",
            "ever_smoked_100_cigs",
            "current_smoking_status",
            "days_smoked_past_month",
            "cigs_per_day_smoking_days",
            "quit_attempt_last_year",
        ]
    ]


def clean_healthcare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean HUQ_J healthcare utilization file.
    Includes general self-rated health, visit frequency, and hospitalization.
    """
    df = df.copy()
    df = df.replace(NHANES_MISSING_CODES, np.nan)

    df["general_health"] = pd.to_numeric(df.get("HUQ010"), errors="coerce")
    df["routine_care_place"] = pd.to_numeric(df.get("HUQ030"), errors="coerce")
    df["healthcare_visits_code"] = pd.to_numeric(df.get("HUQ051"), errors="coerce")
    df["hospitalized_last_year"] = (df.get("HUQ071", pd.Series(2, index=df.index)) == 1).astype(int)

    return df[
        [
            "SEQN",
            "general_health",
            "routine_care_place",
            "healthcare_visits_code",
            "hospitalized_last_year",
        ]
    ]


def clean_insurance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean HIQ_J insurance file.
    Includes coverage and any gap in insurance during the past year.
    """
    df = df.copy()
    df = df.replace(NHANES_MISSING_CODES, np.nan)

    df["insured"] = (df.get("HIQ011", pd.Series(2, index=df.index)) == 1).astype(int)
    df["insurance_gap_last_year"] = (df.get("HIQ210", pd.Series(2, index=df.index)) == 1).astype(int)

    return df[["SEQN", "insured", "insurance_gap_last_year"]]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple mean/mode imputation for remaining missing values.
    Note: for a published study, use multiple imputation (MICE).
    The choice of imputation strategy is documented here intentionally.
    """
    continuous_cols = [
        "age", "poverty_ratio", "met_min_week",
        "sleep_hours_avg", "bmi", "drinks_per_day",
        "drinks_per_week_est", "drink_frequency", "sedentary_minutes",
        "sleep_apnea_symptom_freq", "daytime_sleepiness_freq",
        "current_smoking_status", "days_smoked_past_month",
        "cigs_per_day_smoking_days", "general_health", "routine_care_place",
        "healthcare_visits_code",
    ]
    categorical_cols = ["education", "race_eth", "marital_status", "born_us"]

    for col in continuous_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    print("Loading NHANES XPT files...")

    dpq = load_xpt("P_DPQ.XPT")
    demo = load_xpt("P_DEMO.XPT")
    paq = load_xpt("P_PAQ.XPT")
    slq = load_xpt("P_SLQ.XPT")
    bmx = load_xpt("P_BMX.XPT")
    alq = load_xpt("P_ALQ.XPT")
    smq = load_xpt("P_SMQ.XPT")
    huq = load_xpt("P_HUQ.XPT")
    hiq = load_xpt("P_HIQ.XPT")

    print("Cleaning individual files...")
    phq_clean = clean_phq9(dpq)
    demo_clean = clean_demographics(demo)
    pa_clean = clean_physical_activity(paq)
    sleep_clean = clean_sleep(slq)
    bmi_clean = clean_bmi(bmx)
    alc_clean = clean_alcohol(alq)
    smoking_clean = clean_smoking(smq)
    healthcare_clean = clean_healthcare(huq)
    insurance_clean = clean_insurance(hiq)

    print("Merging on SEQN (participant ID)...")
    df = demo_clean.copy()
    for other in [
        phq_clean,
        pa_clean,
        sleep_clean,
        bmi_clean,
        alc_clean,
        smoking_clean,
        healthcare_clean,
        insurance_clean,
    ]:
        df = df.merge(other, on="SEQN", how="left")

    # Drop participants with no PHQ-9 score (primary outcome missing)
    n_before = len(df)
    df = df.dropna(subset=["phq9_score"])
    n_after = len(df)
    print(f"  Dropped {n_before - n_after} rows with missing PHQ-9 (outcome).")

    # Impute remaining missing predictors
    df = impute_missing(df)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "nhanes_clean.csv"
    df.to_csv(out_path, index=False)

    print(f"\nClean dataset saved: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"\nDepression outcome distribution:")
    print(df["phq9_severity"].value_counts())
    print(f"\nBinary outcome (PHQ-9 >= 10): {df['depression_binary'].mean():.1%} positive")


if __name__ == "__main__":
    main()
