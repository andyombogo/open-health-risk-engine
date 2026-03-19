"""
Clean Streamlit entrypoint for the deployed app.
"""

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO_URL = os.getenv("PROJECT_REPO_URL", "").strip()

st.set_page_config(
    page_title="Open Health Risk Engine",
    page_icon="OH",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(14, 116, 144, 0.18) 0, transparent 28%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.16) 0, transparent 24%),
            linear-gradient(180deg, #eaf4ff 0%, #f7fbff 22%, #ffffff 58%, #f8fafc 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(234, 244, 255, 0.88);
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .hero-card {
        background:
            radial-gradient(circle at top right, rgba(96, 165, 250, 0.35) 0, transparent 28%),
            linear-gradient(135deg, #082f49 0%, #0f172a 52%, #1d4ed8 100%);
        border: 1px solid rgba(191, 219, 254, 0.35);
        border-radius: 28px;
        padding: 1.7rem 1.8rem;
        box-shadow: 0 28px 60px rgba(15, 23, 42, 0.22);
    }
    .hero-kicker {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        color: #dbeafe;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        border: 1px solid rgba(219, 234, 254, 0.16);
    }
    .hero-title {
        color: #f8fafc;
        font-size: clamp(2.2rem, 4vw, 3.3rem);
        font-weight: 800;
        line-height: 1.02;
        margin: 0.95rem 0 0.65rem 0;
        letter-spacing: -0.04em;
    }
    .hero-subcopy {
        color: #dbeafe;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 52rem;
    }
    .hero-meta {
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }
    .hero-chip {
        background: rgba(255, 255, 255, 0.12);
        color: #eff6ff;
        border: 1px solid rgba(219, 234, 254, 0.14);
        border-radius: 999px;
        padding: 0.42rem 0.78rem;
        font-size: 0.83rem;
        font-weight: 600;
    }
    .disclaimer {
        background: rgba(255, 251, 235, 0.92);
        border: 1px solid #fcd34d;
        border-left: 6px solid #d97706;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        font-size: 0.9rem;
        line-height: 1.55;
        color: #1f2937;
        box-shadow: 0 12px 28px rgba(217, 119, 6, 0.1);
    }
    .disclaimer strong {
        color: #92400e;
    }
    .section-shell {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #dbeafe;
        border-radius: 22px;
        padding: 1rem 1rem 0.7rem 1rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }
    .panel-title {
        color: #0f172a;
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .panel-copy {
        color: #475569;
        font-size: 0.94rem;
        margin-bottom: 0.9rem;
    }
    .score-shell {
        background:
            radial-gradient(circle at top right, rgba(125, 211, 252, 0.2) 0, transparent 26%),
            linear-gradient(160deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 24px;
        padding: 1.35rem;
        border: 1px solid #cbd5e1;
        box-shadow: 0 22px 50px rgba(15, 23, 42, 0.1);
    }
    .score-caption {
        color: #475569;
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .metric-chip {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        height: 100%;
    }
    .metric-chip-label {
        color: #64748b;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .metric-chip-value {
        color: #0f172a;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .factor-pill {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        border-radius: 999px;
        padding: 0.4rem 0.75rem;
        display: inline-block;
        margin: 0.25rem 0.35rem 0 0;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .footer-note {
        color: #475569;
        font-size: 0.88rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    try:
        from src.predict_risk import RiskPredictor

        return RiskPredictor()
    except FileNotFoundError:
        return None


predictor = load_predictor()

education_map = {
    "Less than 9th grade": 1,
    "9th-11th grade": 2,
    "High school / GED": 3,
    "Some college": 4,
    "College graduate or above": 5,
}


def format_factor(feature_name: str) -> str:
    factor_labels = {
        "inactive": "No physical activity",
        "short_sleep": "Short sleep duration",
        "sleep_trouble": "Sleep problems",
        "poverty_low": "Below poverty line",
        "hazardous_drinking": "Hazardous alcohol use",
        "obese": "Obesity",
        "met_min_week": "Activity level",
        "sleep_hours": "Sleep duration",
        "bmi": "BMI",
        "drinks_per_week": "Alcohol intake",
        "age": "Age",
        "poverty_ratio": "Income level",
        "education": "Education level",
    }
    return factor_labels.get(feature_name, feature_name.replace("_", " ").title())


def score_inputs(inputs: dict) -> dict:
    if predictor is None:
        return {
            "risk_score": 0.0,
            "risk_label": "Unavailable",
            "risk_color": "gray",
            "phq9_estimate": 0.0,
            "top_factors": [],
        }
    return predictor.predict(inputs)

title_col, repo_col = st.columns([4, 1])
with title_col:
    st.markdown(
        """
<div class="hero-card">
    <div class="hero-kicker">Clinical Calculator Demo</div>
    <div class="hero-title">Open Health Risk Engine</div>
    <div class="hero-subcopy">
        Explainable preventive mental health analytics in a faster, clearer calculator interface.
        Adjust the inputs below and watch the estimated risk update in place.
    </div>
    <div class="hero-meta">
        <span class="hero-chip">Auto-updating calculator</span>
        <span class="hero-chip">NHANES-based demo model</span>
        <span class="hero-chip">Explainable output</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
with repo_col:
    if REPO_URL:
        st.link_button("View GitHub", REPO_URL, use_container_width=True)

st.markdown(
    """
<div class="disclaimer">
<strong>Research tool only.</strong> This is not a diagnostic instrument.
Predictions are based on population-level NHANES data and do not account for
individual clinical history. Do not use this tool to make clinical decisions.
If you are concerned about mental health, please consult a qualified healthcare professional.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")


@st.fragment
def render_calculator():
    input_col, result_col = st.columns([1.15, 0.85], gap="large")

    with input_col:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Calculator Inputs</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Adjust the fields below and the score updates automatically.</div>',
            unsafe_allow_html=True,
        )

        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            age = st.slider("Age", 18, 80, 35, key="calc_age")
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True, key="calc_sex")
            poverty_ratio = st.slider(
                "Poverty-Income Ratio",
                0.0,
                5.0,
                2.5,
                0.1,
                key="calc_poverty_ratio",
                help="Household income divided by the poverty threshold.",
            )
            bmi = st.slider("BMI (kg/m^2)", 15.0, 50.0, 24.0, 0.5, key="calc_bmi")

        with demo_col2:
            education = st.selectbox(
                "Education",
                list(education_map.keys()),
                index=3,
                key="calc_education",
            )
            sleep_hours = st.slider(
                "Sleep (hours/night)",
                3.0,
                12.0,
                7.0,
                0.5,
                key="calc_sleep_hours",
            )
            drinks_per_week = st.slider(
                "Drinks per week",
                0,
                40,
                3,
                key="calc_drinks_per_week",
            )
            sleep_trouble = st.toggle(
                "Regular trouble sleeping",
                value=False,
                key="calc_sleep_trouble",
            )

        met_min_week = st.slider(
            "Weekly Physical Activity (MET-min/week)",
            0,
            3000,
            300,
            50,
            key="calc_met_min_week",
            help="WHO guidance is about 600 MET-min/week or more.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        inputs = {
            "age": age,
            "sex_female": 1 if sex == "Female" else 0,
            "poverty_ratio": poverty_ratio,
            "met_min_week": met_min_week,
            "sleep_hours": sleep_hours,
            "sleep_trouble": int(sleep_trouble),
            "bmi": bmi,
            "drinks_per_week": drinks_per_week,
            "education": education_map[education],
            "race_eth": 3,
        }
        result = score_inputs(inputs)

    with result_col:
        risk_score = result["risk_score"]
        risk_label = result["risk_label"]
        risk_color = result["risk_color"]
        phq9_est = result["phq9_estimate"]
        top_factors = result["top_factors"]

        color_map = {
            "green": "#059669",
            "blue": "#2563eb",
            "orange": "#d97706",
            "red": "#dc2626",
            "darkred": "#7f1d1d",
            "gray": "#64748b",
        }
        hex_color = color_map.get(risk_color, "#64748b")

        st.markdown('<div class="score-shell">', unsafe_allow_html=True)
        st.markdown('<div class="score-caption">Estimated Risk</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div style="font-size:3.25rem; font-weight:800; color:{hex_color}; line-height:1; margin-top:0.35rem;">
    {risk_score:.0%}
</div>
<div style="font-size:1.1rem; font-weight:700; color:{hex_color}; margin-top:0.45rem;">
    {risk_label}
</div>
<div style="font-size:0.95rem; color:#475569; margin-top:0.6rem;">
    Estimated PHQ-9 equivalent: <strong>{phq9_est}</strong>
</div>
""",
            unsafe_allow_html=True,
        )
        st.progress(risk_score)

        chip_col1, chip_col2 = st.columns(2)
        with chip_col1:
            st.markdown(
                f"""
<div class="metric-chip">
    <div class="metric-chip-label">Activity</div>
    <div class="metric-chip-value">{met_min_week} MET-min/week</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with chip_col2:
            st.markdown(
                f"""
<div class="metric-chip">
    <div class="metric-chip-label">Sleep</div>
    <div class="metric-chip-value">{sleep_hours:.1f} hrs/night</div>
</div>
""",
                unsafe_allow_html=True,
            )

        chip_col3, chip_col4 = st.columns(2)
        with chip_col3:
            st.markdown(
                f"""
<div class="metric-chip">
    <div class="metric-chip-label">BMI</div>
    <div class="metric-chip-value">{bmi:.1f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with chip_col4:
            st.markdown(
                f"""
<div class="metric-chip">
    <div class="metric-chip-label">Alcohol</div>
    <div class="metric-chip-value">{drinks_per_week} / week</div>
</div>
""",
                unsafe_allow_html=True,
            )

        if top_factors:
            st.markdown(
                '<div class="score-caption" style="margin-top:1rem;">Top Drivers</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "".join(
                    f'<span class="factor-pill">{format_factor(factor["feature"])}</span>'
                    for factor in top_factors[:5]
                ),
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.caption(
            f"{'Meets' if met_min_week >= 600 else 'Below'} WHO activity guidance"
        )
    with info_col2:
        st.caption(f"{'Optimal' if 7 <= sleep_hours <= 9 else 'Suboptimal'} sleep window")
    with info_col3:
        st.caption(
            "Auto-compute is active. Each change reruns only this calculator block."
        )


render_calculator()

with st.expander("How to interpret this result"):
    st.markdown(
        """
- This score estimates depression risk from lifestyle and demographic patterns seen in NHANES survey data.
- It is designed as a portfolio demo and explainable ML example, not as a diagnostic tool.
- Higher scores suggest a profile that looks more similar to the higher-risk patterns learned by the model.
"""
    )

with st.expander("Model snapshot"):
    st.markdown(
        """
- Dataset: NHANES 2017-March 2020 pre-pandemic
- Deployment model: Random Forest
- Validation: 5-fold stratified cross-validation
- Explainability: feature importance surfaced for the current estimate
"""
    )

st.markdown("---")
footer_repo = f"[GitHub repository]({REPO_URL})" if REPO_URL else "GitHub repository"
st.markdown(
    f'<div class="footer-note">Open Health Risk Engine | NHANES-based research demo | {footer_repo}</div>',
    unsafe_allow_html=True,
)
