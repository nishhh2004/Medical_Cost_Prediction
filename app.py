import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model  = pickle.load(open('medical_cost_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page config
st.set_page_config(
    page_title="Medical Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #0d0d0d;
    font-family: 'Space Grotesk', sans-serif;
    color: #f0f0f0;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 2rem 2rem;
    max-width: 1300px;
}

/* TOP NAV BAR */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 2rem;
    border-bottom: 1px solid #1f1f1f;
    margin-bottom: 3rem;
}
.nav-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 3px;
    color: #ff6b00;
}
.nav-tags {
    display: flex;
    gap: 10px;
}
.nav-tag {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #888;
    padding: 5px 14px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.nav-tag.active {
    border-color: #ff6b00;
    color: #ff6b00;
    background: rgba(255,107,0,0.08);
}

/* PAGE TITLE */
.page-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 7vw, 6rem);
    letter-spacing: 4px;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.page-title span {
    color: #ff6b00;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #555;
    font-weight: 400;
    margin-bottom: 3rem;
    letter-spacing: 0.3px;
}

/* STATS ROW */
.stats-row {
    display: flex;
    gap: 1px;
    margin-bottom: 3rem;
    background: #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1f1f1f;
}
.stat-block {
    flex: 1;
    padding: 1.2rem 1.5rem;
    background: #111;
    text-align: center;
    transition: background 0.2s;
}
.stat-block:hover { background: #161616; }
.stat-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #ff6b00;
    letter-spacing: 2px;
}
.stat-key {
    font-size: 0.7rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 2px;
}

/* SECTION LABEL */
.section-label {
    font-size: 0.7rem;
    color: #ff6b00;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 600;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1f1f1f;
}

/* INPUT CARD */
.input-card {
    background: #111111;
    border: 1px solid #1f1f1f;
    border-radius: 14px;
    padding: 1.8rem;
    height: 100%;
    transition: border-color 0.3s;
}
.input-card:hover {
    border-color: #2a2a2a;
}
.input-card-title {
    font-size: 0.65rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1a1a1a;
}

/* Streamlit widget overrides */
.stSlider label, .stSelectbox label {
    color: #666 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
div[data-baseweb="select"] > div {
    background: #0d0d0d !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    color: #f0f0f0 !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #ff6b00 !important;
}
.stSlider > div > div > div {
    background: linear-gradient(90deg, #ff6b00, #ff9500) !important;
}
.stSlider > div > div > div > div {
    background: #ff6b00 !important;
    border: 2px solid #fff !important;
    box-shadow: 0 0 10px rgba(255,107,0,0.5) !important;
}

/* PREDICT BUTTON */
.stButton > button {
    width: 100%;
    background: #ff6b00;
    color: #000;
    border: none;
    border-radius: 10px;
    padding: 0.9rem 2rem;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 3px;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-top: 1.5rem;
}
.stButton > button:hover {
    background: #ff8c00;
    transform: translateY(-1px);
    box-shadow: 0 6px 25px rgba(255,107,0,0.35);
    color: #000;
}

/* RESULT */
.result-wrapper {
    background: #111;
    border: 1px solid #1f1f1f;
    border-top: 3px solid #ff6b00;
    border-radius: 14px;
    padding: 2.5rem;
    margin: 2rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.result-left {}
.result-tag {
    font-size: 0.65rem;
    color: #ff6b00;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.5rem;
}
.result-amount {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.5rem, 8vw, 5.5rem);
    color: #ffffff;
    letter-spacing: 3px;
    line-height: 1;
}
.result-note {
    font-size: 0.75rem;
    color: #444;
    margin-top: 0.5rem;
}
.result-right {
    text-align: right;
}
.result-verdict {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
.verdict-high { color: #ff4444; }
.verdict-mid  { color: #ff9500; }
.verdict-low  { color: #44ff88; }

/* RISK CARDS */
.risk-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}
.risk-item {
    flex: 1;
    min-width: 200px;
    background: #111;
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}
.risk-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.dot-red    { background: #ff4444; box-shadow: 0 0 8px #ff4444; }
.dot-green  { background: #44ff88; box-shadow: 0 0 8px #44ff88; }
.dot-orange { background: #ff9500; box-shadow: 0 0 8px #ff9500; }
.risk-content {}
.risk-name {
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.risk-status {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 2px;
    color: #f0f0f0;
}
.risk-desc {
    font-size: 0.75rem;
    color: #444;
    margin-top: 3px;
    line-height: 1.5;
}

/* SUMMARY PILLS */
.pills-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1rem;
}
.pill {
    background: #161616;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 0.78rem;
    color: #888;
    display: flex;
    gap: 6px;
    align-items: center;
}
.pill b { color: #f0f0f0; font-weight: 500; }

/* METRICS BOTTOM */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 120px;
    background: #111;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: #ff6b00;
    letter-spacing: 1px;
}
.metric-key {
    font-size: 0.65rem;
    color: #333;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 2px;
}

/* FOOTER */
.footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a1a1a;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}
.footer-left {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 2px;
    color: #333;
}
.footer-right {
    font-size: 0.72rem;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-logo">MEDICOST</div>
    <div class="nav-tags">
        <div class="nav-tag">Linear Regression</div>
        <div class="nav-tag">Scikit-Learn</div>
        <div class="nav-tag">Streamlit</div>
        <div class="nav-tag active">● Live</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── PAGE TITLE ────────────────────────────────────────────────
st.markdown("""
<div class="page-title">MEDICAL <span>COST</span> PREDICTOR</div>
<div class="page-subtitle">
    Enter your details below — the model will estimate your annual insurance charges.
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ─────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-block">
        <div class="stat-val">78%</div>
        <div class="stat-key">R² Score</div>
    </div>
    <div class="stat-block">
        <div class="stat-val">1338</div>
        <div class="stat-key">Training Records</div>
    </div>
    <div class="stat-block">
        <div class="stat-val">±$4186</div>
        <div class="stat-key">Avg Error (MAE)</div>
    </div>
    <div class="stat-block">
        <div class="stat-val">6</div>
        <div class="stat-key">Input Features</div>
    </div>
    <div class="stat-block">
        <div class="stat-val">OLS</div>
        <div class="stat-key">Algorithm</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT SECTION ─────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Input</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="input-card"><div class="input-card-title">Personal Info</div>',
                unsafe_allow_html=True)
    age      = st.slider("Age", 18, 64, 30)
    sex      = st.selectbox("Sex", ["Male", "Female"])
    children = st.selectbox("Children / Dependents", [0, 1, 2, 3, 4, 5])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card"><div class="input-card-title">Health Info</div>',
                unsafe_allow_html=True)
    bmi    = st.slider("BMI", 15.0, 55.0, 25.0, step=0.1)
    smoker = st.selectbox("Smoker", ["No", "Yes"])

    if bmi < 18.5:
        bmi_cat, bmi_color = "Underweight", "#38bdf8"
    elif bmi < 25:
        bmi_cat, bmi_color = "Normal Weight", "#44ff88"
    elif bmi < 30:
        bmi_cat, bmi_color = "Overweight", "#ff9500"
    else:
        bmi_cat, bmi_color = "Obese", "#ff4444"

    bmi_pct = int(((bmi - 15) / (55 - 15)) * 100)
    st.markdown(f"""
    <div style="margin-top:1.2rem; padding:1rem;
                background:#0d0d0d; border-radius:10px;
                border:1px solid #1a1a1a;">
        <div style="font-size:0.65rem; color:#444;
                    text-transform:uppercase; letter-spacing:1.5px;
                    margin-bottom:6px;">BMI Category</div>
        <div style="font-family:'Bebas Neue',sans-serif;
                    font-size:1.3rem; letter-spacing:2px;
                    color:{bmi_color};">{bmi_cat}</div>
        <div style="height:4px; background:#1a1a1a;
                    border-radius:4px; margin-top:10px; 
                    position:relative; overflow:hidden;">
            <div style="position:absolute; left:0; top:0;
                        height:100%; width:{bmi_pct}%;
                        background:linear-gradient(90deg,#44ff88,#ff9500,#ff4444);
                        border-radius:4px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between;
                    font-size:0.62rem; color:#333; margin-top:5px;">
            <span>15</span><span>BMI Scale</span><span>55</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-card"><div class="input-card-title">Location & Summary</div>',
                unsafe_allow_html=True)
    region = st.selectbox("US Region",
                          ["Northeast", "Northwest",
                           "Southeast", "Southwest"])
    st.markdown(f"""
    <div class="pills-wrap">
        <div class="pill">Age <b>{age} yrs</b></div>
        <div class="pill">Sex <b>{sex}</b></div>
        <div class="pill">BMI <b>{bmi}</b></div>
        <div class="pill">Kids <b>{children}</b></div>
        <div class="pill">Smoker <b>{smoker}</b></div>
        <div class="pill">Region <b>{region}</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("CALCULATE CHARGES")

# ── RESULTS ───────────────────────────────────────────────────
if predict:
    sex_encoded    = 1 if sex == "Male" else 0
    smoker_encoded = 1 if smoker == "Yes" else 0
    region_encoded = {
        "Northeast": 0, "Northwest": 1,
        "Southeast": 2, "Southwest": 3
    }[region]

    input_data   = np.array([[age, sex_encoded, bmi,
                               children, smoker_encoded, region_encoded]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]

    # Verdict
    if prediction > 30000:
        verdict       = "HIGH RISK"
        verdict_class = "verdict-high"
    elif prediction > 15000:
        verdict       = "MODERATE RISK"
        verdict_class = "verdict-mid"
    else:
        verdict       = "LOW RISK"
        verdict_class = "verdict-low"

    st.markdown('<div class="section-label">Prediction Result</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-wrapper">
        <div class="result-left">
            <div class="result-tag">Estimated Annual Charges</div>
            <div class="result-amount">${prediction:,.0f}</div>
            <div class="result-note">Margin of error ≈ ±$4,186 · Linear Regression · R² = 0.78</div>
        </div>
        <div class="result-right">
            <div style="font-size:0.65rem; color:#444;
                        text-transform:uppercase; letter-spacing:2px;
                        margin-bottom:6px;">Risk Level</div>
            <div class="result-verdict {verdict_class}">{verdict}</div>
            <div style="font-size:0.72rem; color:#333; margin-top:4px;">
                Based on smoker, BMI & age factors
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk breakdown
    st.markdown('<div class="section-label">Risk Breakdown</div>',
                unsafe_allow_html=True)

    smoke_status = ("HIGH — Smokers pay ≈3–4x more",
                    "dot-red") if smoker == "Yes" else \
                   ("LOW — Non-smoker advantage", "dot-green")

    bmi_status   = ("HIGH — BMI above 30 is obese range",
                    "dot-red") if bmi >= 30 else \
                   ("MODERATE — Slightly above normal",
                    "dot-orange") if bmi >= 25 else \
                   ("LOW — BMI in healthy range", "dot-green")

    age_status   = ("HIGH — Above 50 raises charges significantly",
                    "dot-red") if age >= 50 else \
                   ("MODERATE — Middle age bracket",
                    "dot-orange") if age >= 35 else \
                   ("LOW — Young age, lower risk", "dot-green")

    st.markdown(f"""
    <div class="risk-row">
        <div class="risk-item">
            <div class="risk-dot {smoke_status[1]}"></div>
            <div class="risk-content">
                <div class="risk-name">Smoking</div>
                <div class="risk-status">{smoke_status[0].split('—')[0].strip()}</div>
                <div class="risk-desc">{smoke_status[0].split('—')[1].strip()}</div>
            </div>
        </div>
        <div class="risk-item">
            <div class="risk-dot {bmi_status[1]}"></div>
            <div class="risk-content">
                <div class="risk-name">BMI</div>
                <div class="risk-status">{bmi_status[0].split('—')[0].strip()}</div>
                <div class="risk-desc">{bmi_status[0].split('—')[1].strip()}</div>
            </div>
        </div>
        <div class="risk-item">
            <div class="risk-dot {age_status[1]}"></div>
            <div class="risk-content">
                <div class="risk-name">Age</div>
                <div class="risk-status">{age_status[0].split('—')[0].strip()}</div>
                <div class="risk-desc">{age_status[0].split('—')[1].strip()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model metrics
    st.markdown('<div class="section-label" style="margin-top:2rem;">Model Performance</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="metrics-row">
        <div class="metric-box">
            <div class="metric-val">$4,186</div>
            <div class="metric-key">MAE</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">$5,799</div>
            <div class="metric-key">RMSE</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">0.78</div>
            <div class="metric-key">R² Score</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">1338</div>
            <div class="metric-key">Train Size</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">80/20</div>
            <div class="metric-key">Split Ratio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">MEDICOST · 2025</div>
    <div class="footer-right">
        Linear Regression · Scikit-Learn · Streamlit · 
        Dataset: Medical Cost Personal Dataset
    </div>
</div>
""", unsafe_allow_html=True)