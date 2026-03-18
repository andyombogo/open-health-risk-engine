"""
dashboard/app.py
-----------------
Streamlit web application for the Open Health Risk Engine.

Run with:
    streamlit run dashboard/app.py

Features:
  - Interactive risk scoring from lifestyle inputs
  - Real-time SHAP explanation of the prediction
  - Risk factor visualization
  - Population context (how this person compares to NHANES sample)
  - Ethical disclaimer and clinical context
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO_URL = os.getenv("PROJECT_REPO_URL", "").strip()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Open Health Risk Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .factor-bar {
        height: 8px;
        border-radius: 4px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    try:
        from src.predict_risk import RiskPredictor
        return RiskPredictor()
    except FileNotFoundError:
        return None


predictor = load_predictor()

# ── Sidebar: Input form ───────────────────────────────────────────────────────
st.sidebar.title("🧠 Health Risk Engine")
st.sidebar.markdown("*Enter lifestyle and demographic information below.*")
st.sidebar.divider()

st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age", 18, 80, 35)
sex = st.sidebar.radio("Sex", ["Male", "Female"], horizontal=True)
sex_female = 1 if sex == "Female" else 0

education_map = {
    "Less than 9th grade": 1,
    "9th–11th grade": 2,
    "High school / GED": 3,
    "Some college": 4,
    "College graduate or above": 5,
}
education = st.sidebar.selectbox("Education", list(education_map.keys()), index=3)
education_val = education_map[education]

poverty_ratio = st.sidebar.slider(
    "Poverty-Income Ratio",
    0.0, 5.0, 2.5, 0.1,
    help="Household income ÷ poverty threshold. <1.0 = below poverty line."
)

st.sidebar.divider()
st.sidebar.subheader("Physical Activity")
met_min_week = st.sidebar.slider(
    "Weekly Physical Activity (MET-min/week)",
    0, 3000, 300, 50,
    help="WHO recommends ≥600 MET-min/week (e.g. 150 min moderate activity)."
)

st.sidebar.divider()
st.sidebar.subheader("Sleep")
sleep_hours = st.sidebar.slider("Average Sleep (hours/night)", 3.0, 12.0, 7.0, 0.5)
sleep_trouble = st.sidebar.checkbox("Regularly trouble sleeping?")

st.sidebar.divider()
st.sidebar.subheader("Body Metrics")
bmi = st.sidebar.slider("BMI (kg/m²)", 15.0, 50.0, 24.0, 0.5)

st.sidebar.divider()
st.sidebar.subheader("Alcohol")
drinks_per_week = st.sidebar.slider("Drinks per week (estimated)", 0, 40, 3)

# ── Main content ──────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("Open Health Risk Engine")
    st.markdown("**Explainable ML for Preventive Mental Health Analytics**")
    if REPO_URL:
        st.caption(f"[View the GitHub repository]({REPO_URL})")

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Research tool only.</strong> This is not a diagnostic instrument. 
Predictions are based on population-level NHANES data and do not account for 
individual clinical history. Do not use this tool to make clinical decisions. 
If you are concerned about mental health, please consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
inputs = {
    "age": age,
    "sex_female": sex_female,
    "poverty_ratio": poverty_ratio,
    "met_min_week": met_min_week,
    "sleep_hours": sleep_hours,
    "sleep_trouble": int(sleep_trouble),
    "bmi": bmi,
    "drinks_per_week": drinks_per_week,
    "education": education_val,
    "race_eth": 3,  # Default; could add to sidebar
}

col_score, col_factors, col_context = st.columns([1, 1, 1])

# ── Column 1: Risk Score ──────────────────────────────────────────────────────
with col_score:
    st.subheader("Risk Score")

    if predictor is None:
        st.warning("Model not loaded. Run `python src/train_model.py` first.")
        risk_score = 0.0
        risk_label = "—"
        risk_color = "gray"
        phq9_est = 0.0
        top_factors = []
    else:
        result = predictor.predict(inputs)
        risk_score = result["risk_score"]
        risk_label = result["risk_label"]
        risk_color = result["risk_color"]
        phq9_est = result["phq9_estimate"]
        top_factors = result["top_factors"]

    # Color mapping for Streamlit
    color_map = {
        "green": "#059669",
        "blue": "#2563EB",
        "orange": "#D97706",
        "red": "#DC2626",
        "darkred": "#7F1D1D",
        "gray": "#6B7280",
    }
    hex_color = color_map.get(risk_color, "#6B7280")

    # Score gauge (simple progress bar + label)
    st.markdown(f"""
    <div style="background:{hex_color}15; border: 2px solid {hex_color}; 
                border-radius:12px; padding:1.5rem; text-align:center;">
        <div style="font-size:3rem; font-weight:700; color:{hex_color}">
            {risk_score:.0%}
        </div>
        <div style="font-size:1.1rem; font-weight:600; color:{hex_color}; margin-top:0.25rem">
            {risk_label}
        </div>
        <div style="font-size:0.8rem; color:#6b7280; margin-top:0.5rem">
            Estimated PHQ-9: {phq9_est}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(risk_score)

    # WHO activity context
    st.markdown("---")
    who_met = 600
    activity_pct = min(met_min_week / who_met, 1.0)
    who_color = "#059669" if met_min_week >= who_met else "#DC2626"
    st.markdown(f"<div class='metric-label'>Physical activity vs. WHO recommendation</div>",
                unsafe_allow_html=True)
    st.progress(activity_pct)
    st.caption(
        f"{'✅ Meets' if met_min_week >= who_met else '❌ Below'} WHO guidelines "
        f"({met_min_week} / {who_met} MET-min/week)"
    )

    # Sleep context
    sleep_ok = 7 <= sleep_hours <= 9
    st.caption(
        f"{'✅ Optimal' if sleep_ok else '⚠️ Suboptimal'} sleep: {sleep_hours} hrs/night "
        f"(recommended: 7–9 hrs)"
    )

# ── Column 2: Risk Factors ────────────────────────────────────────────────────
with col_factors:
    st.subheader("Key Risk Factors")
    st.caption("Top predictors from the model for this individual")

    if top_factors:
        factor_labels = {
            "inactive": "No physical activity",
            "short_sleep": "Short sleep duration",
            "sleep_trouble": "Sleep problems",
            "poverty_low": "Below poverty line",
            "hazardous_drinking": "Hazardous alcohol use",
            "obese": "Obesity (BMI≥30)",
            "met_min_week": "Physical activity level",
            "sleep_hours": "Sleep hours",
            "bmi": "BMI",
            "drinks_per_week": "Alcohol consumption",
            "age": "Age",
            "poverty_ratio": "Income level",
            "education": "Education level",
        }

        for i, factor in enumerate(top_factors):
            feat = factor["feature"]
            imp = factor["importance"]
            label = factor_labels.get(feat, feat.replace("_", " ").title())

            bar_width = int(imp / max(f["importance"] for f in top_factors) * 100)
            bar_color = "#DC2626" if i < 3 else "#2563EB"

            st.markdown(f"""
            <div style="margin-bottom: 0.75rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                    <span>{label}</span>
                    <span style="color:#6b7280">importance: {imp:.3f}</span>
                </div>
                <div style="background:#f3f4f6; border-radius:4px; height:8px; margin-top:4px;">
                    <div style="background:{bar_color}; width:{bar_width}%; 
                                height:8px; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Train the model to see risk factors.")

# ── Column 3: Context ─────────────────────────────────────────────────────────
with col_context:
    st.subheader("Population Context")
    st.caption("How this profile compares to the NHANES population")

    # Summary table of inputs
    context_data = {
        "Factor": ["Age", "BMI", "Activity", "Sleep", "Drinks/week", "Income ratio"],
        "Your value": [
            f"{age} yrs",
            f"{bmi:.1f}",
            f"{met_min_week} MET-min/wk",
            f"{sleep_hours:.1f} hrs",
            str(drinks_per_week),
            f"{poverty_ratio:.1f}",
        ],
        "Population avg": [
            "~47 yrs",
            "~29.0",
            "~450 MET-min/wk",
            "~7.0 hrs",
            "~4.5",
            "~2.8",
        ],
    }
    st.dataframe(pd.DataFrame(context_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**PHQ-9 Severity Scale**")
    severity_data = {
        "Score": ["0–4", "5–9", "10–14", "15–19", "20–27"],
        "Severity": ["Minimal", "Mild", "Moderate", "Mod. severe", "Severe"],
    }
    st.dataframe(pd.DataFrame(severity_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    **About this model**
    - Dataset: NHANES 2017-March 2020 pre-pandemic
    - Algorithm: Random Forest (300 trees)
    - Validation: 5-fold stratified CV
    - Explainability: SHAP values
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.8rem;">
    Open Health Risk Engine · Built with NHANES data, scikit-learn, and SHAP · 
    <a href="https://github.com/andyombogo/open-health-risk-engine" 
       style="color:#6b7280">GitHub</a> · MIT License · Research use only
</div>
""", unsafe_allow_html=True)
