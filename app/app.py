import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from granite_helper import get_granite_response

APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.resolve()
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #E0F2F1;
}

.stApp {
    background: linear-gradient(135deg, #0a1f1a 0%, #0d2b24 50%, #0a1f1a 100%);
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2b24 0%, #1a3a32 100%) !important;
    border-right: 1px solid #00D9FF33;
}

section[data-testid="stSidebar"] * {
    color: #E0F2F1 !important;
}

.hero-banner {
    background: linear-gradient(135deg, #00D9FF15, #00E5B315);
    border: 1px solid #00D9FF44;
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 24px;
    text-align: center;
}

.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00D9FF, #00E5B3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
    line-height: 1.2;
}

.hero-sub {
    font-size: 1.15rem;
    color: #80CBC4;
    max-width: 600px;
    margin: 0 auto;
}

.metric-card {
    background: linear-gradient(135deg, #1a3a32, #0d2b24);
    border: 1px solid #00D9FF44;
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    transition: transform 0.2s;
}

.metric-card:hover { transform: translateY(-4px); border-color: #00D9FF99; }

.metric-card .val {
    font-size: 2rem;
    font-weight: 900;
    color: #00D9FF;
    display: block;
}

.metric-card .lbl {
    font-size: 0.85rem;
    color: #80CBC4;
    margin-top: 4px;
    display: block;
}

.step-card {
    background: #1a3a3288;
    border: 1px solid #00E5B344;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    height: 100%;
}

.step-num {
    font-size: 2rem;
    font-weight: 900;
    color: #00E5B3;
}

.step-title {
    font-size: 1rem;
    font-weight: 700;
    color: #E0F2F1;
    margin: 8px 0 4px;
}

.step-desc { font-size: 0.85rem; color: #80CBC4; }

.prediction-box {
    background: linear-gradient(135deg, #00D9FF22, #00E5B322);
    border: 2px solid #00D9FF;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 16px 0;
}

.pred-label {
    font-size: 2.4rem;
    font-weight: 900;
    color: #00D9FF;
    letter-spacing: 3px;
}

.pred-icon { font-size: 3rem; }

.info-card {
    background: #1a3a3288;
    border-left: 4px solid #00D9FF;
    border-radius: 0 12px 12px 0;
    padding: 18px 20px;
    margin: 12px 0;
}

.info-card h4 {
    color: #00D9FF;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.info-card p {
    color: #B2DFDB;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0;
}

.score-bar-wrap {
    background: #0d2b24;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin: 8px 0;
}

.score-bar {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #00D9FF, #00E5B3);
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #00E5B3;
    margin: 24px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #00E5B333;
}

.sidebar-logo {
    font-size: 1.5rem;
    font-weight: 900;
    color: #00D9FF;
    text-align: center;
    padding: 12px 0 4px;
}

.sidebar-sub {
    font-size: 0.8rem;
    color: #80CBC4;
    text-align: center;
    margin-bottom: 16px;
}

.tag {
    display: inline-block;
    background: #00D9FF22;
    color: #00D9FF;
    border: 1px solid #00D9FF55;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px 2px;
}

.stMetric { background: #1a3a3266; border-radius: 12px; padding: 12px; border: 1px solid #00D9FF33; }
.stMetric label { color: #80CBC4 !important; font-size: 0.85rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #00D9FF !important; font-size: 1.8rem !important; font-weight: 800 !important; }

div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

footer { visibility: hidden; }

.footer-bar {
    background: linear-gradient(90deg, #00D9FF11, #00E5B311);
    border-top: 1px solid #00D9FF33;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    color: #80CBC4;
    font-size: 0.85rem;
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)

WASTE_INFO = {
    "cardboard": {"disposal": "Flatten boxes and remove tape/plastic. Drop at cardboard recycling centre.", "impact": "Recycling 1 ton saves 17 trees and 7,000 gallons of water.", "time": "~60 days", "sdg": "SDG 12", "recommendation": "Reuse boxes first. Compost non-coated cardboard.", "emoji": "📦", "color": "#D4A853"},
    "glass":     {"disposal": "Rinse clean. Remove lids. Separate by colour if required.", "impact": "Glass recycles infinitely. Saves 40% energy vs new glass.", "time": "1,000,000+ years", "sdg": "SDG 12", "recommendation": "Reuse glass jars for storage.", "emoji": "🍾", "color": "#4FC3F7"},
    "metal":     {"disposal": "Rinse cans. Separate aluminium from steel. Mind sharp edges.", "impact": "Recycling aluminium saves 95% energy.", "time": "200–500 years", "sdg": "SDG 12", "recommendation": "Separate aluminium from steel for best results.", "emoji": "🥫", "color": "#90A4AE"},
    "paper":     {"disposal": "Keep dry and clean. Remove plastic windows and coatings.", "impact": "Recycling 1 ton saves 24 trees and 682 gallons of water.", "time": "~5–6 weeks", "sdg": "SDG 12", "recommendation": "Shred confidential docs safely. Use as garden mulch.", "emoji": "📄", "color": "#FFF176"},
    "plastic":   {"disposal": "Check recycling number. Rinse clean. Most sites take #1 and #2.", "impact": "Reduces emissions by 75% vs virgin plastic production.", "time": "400–1,000 years", "sdg": "SDG 12", "recommendation": "Switch to reusable alternatives where possible.", "emoji": "🧴", "color": "#CE93D8"},
    "trash":     {"disposal": "Non-recyclable items go to general waste / landfill.", "impact": "Landfills produce methane, a potent greenhouse gas.", "time": "20–30 years avg", "sdg": "SDG 13", "recommendation": "Minimise trash by choosing recyclable/compostable products.", "emoji": "🗑️", "color": "#EF9A9A"},
}
SUSTAINABILITY_SCORE = {"cardboard": 88, "glass": 95, "metal": 92, "paper": 90, "plastic": 75, "trash": 40}

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model not found at: {MODEL_PATH}"
    if MODEL_PATH.stat().st_size < 1_000_000:
        return None, "Model file too small / corrupted."
    try:
        return YOLO(str(MODEL_PATH)), None
    except Exception as e:
        return None, str(e)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">♻️ EcoSort AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Intelligent Waste Classification</div>', unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio("Navigate", ["🏠 Home", "📸 Predict", "📊 Dashboard", "ℹ️ About"],
                    label_visibility="collapsed", key="nav_mode")
    st.markdown("---")
    st.markdown("**Classifies:**")
    for k, v in WASTE_INFO.items():
        st.markdown(f"{v['emoji']} {k.capitalize()}")
    st.markdown("---")
    st.caption("Powered by YOLOv8 · IBM watsonx")

# ── HOME ──────────────────────────────────────────────────────────────────────
if mode == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Sort Smarter with EcoSort AI</div>
        <p class="hero-sub">Advanced computer vision that classifies waste instantly and guides you toward sustainable disposal — powering the circular economy.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [("24.5K", "Tonnes Sorted"), ("89%", "Recycling Rate"), ("88.4%", "AI Accuracy"), ("6", "Waste Classes")]
    for col, (val, lbl) in zip([c1, c2, c3, c4], cards):
        col.markdown(f'<div class="metric-card"><span class="val">{val}</span><span class="lbl">{lbl}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌱 How It Works</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    steps = [("1", "📸 Capture", "Take or upload a photo of your waste item"),
             ("2", "🤖 Identify", "YOLOv8 AI classifies the material in seconds"),
             ("3", "♻️ Act", "Get disposal guidance, impact data & SDG alignment")]
    for col, (num, title, desc) in zip([s1, s2, s3], steps):
        col.markdown(f'<div class="step-card"><div class="step-num">{num}</div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌍 Why It Matters</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("""
        <div class="info-card">
            <h4>🚨 The Problem</h4>
            <p>2.12 billion tonnes of waste are generated annually. Only 35% is properly managed. Landfills emit methane — a greenhouse gas 84× more potent than CO₂.</p>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="info-card">
            <h4>✅ Our Solution</h4>
            <p>Automated AI classification gives instant disposal guidance, reduces contamination, and empowers individuals to contribute to the circular economy every day.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Sustainability Scores by Material</div>', unsafe_allow_html=True)
    labels = list(SUSTAINABILITY_SCORE.keys())
    scores = list(SUSTAINABILITY_SCORE.values())
    colors = [WASTE_INFO[l]["color"] for l in labels]
    fig = go.Figure(go.Bar(
        x=[l.capitalize() for l in labels], y=scores,
        marker_color=colors, text=scores, textposition="outside",
        textfont=dict(color="#E0F2F1", size=13)
    ))
    fig.update_layout(
        plot_bgcolor="#0d2b24", paper_bgcolor="#0d2b24",
        font=dict(color="#E0F2F1", family="Inter"),
        yaxis=dict(range=[0, 110], gridcolor="#1a3a32", title="Score / 100"),
        xaxis=dict(gridcolor="#1a3a32"),
        margin=dict(l=20, r=20, t=20, b=20), height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PREDICT ───────────────────────────────────────────────────────────────────
elif mode == "📸 Predict":
    st.markdown('<div class="section-title">📸 Waste Classification</div>', unsafe_allow_html=True)
    st.caption("Upload a photo — the AI identifies the waste type and provides full disposal guidance.")

    model, model_err = load_model()
    if model is None:
        st.error(f"❌ Model could not be loaded: {model_err}")
        st.stop()
    else:
        st.success(f"✅ Model ready — `{MODEL_PATH.name}`")

    col1, col2 = st.columns([1, 1], gap="large")
    uploaded_file = None

    with col1:
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a JPG or PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded image")

    label = None
    confidence = 0.0
    all_probs = []

    with col2:
        st.markdown("#### 🤖 AI Prediction")
        if uploaded_file:
            with st.spinner("Analysing…"):
                try:
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format="PNG")
                    temp_path = PROJECT_ROOT / "temp_image.png"
                    temp_path.write_bytes(img_bytes.getvalue())
                    results = model.predict(str(temp_path), verbose=False)
                    top1 = results[0].probs.top1
                    label = results[0].names[int(top1)]
                    confidence = float(results[0].probs.top1conf)
                    all_probs = [(results[0].names[i], float(p)) for i, p in enumerate(results[0].probs.data)]
                    temp_path.unlink(missing_ok=True)
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")

            if label:
                conf_pct = confidence * 100
                if confidence < 0.55:
                    st.markdown(f"""
                    <div class="prediction-box" style="border-color: #EF9A9A; background: rgba(239, 154, 154, 0.1);">
                        <div class="pred-icon">⚠️</div>
                        <div class="pred-label" style="color: #EF9A9A; letter-spacing: 2px;">UNRECOGNIZED</div>
                    </div>""", unsafe_allow_html=True)
                    st.error(f"🎯 Confidence: **{conf_pct:.1f}%** — Too Low (Below 55%)")
                    st.info("💡 **EcoSort AI Hint**: The uploaded image does not match any of our 6 waste categories with high confidence. Please ensure the waste item is clearly visible and belongs to Cardboard, Glass, Metal, Paper, Plastic, or Trash.")
                else:
                    info = WASTE_INFO.get(label.lower(), {})
                    emoji = info.get("emoji", "♻️")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="pred-icon">{emoji}</div>
                        <div class="pred-label">{label.upper()}</div>
                    </div>""", unsafe_allow_html=True)

                    if confidence >= 0.90:
                        st.success(f"🎯 Confidence: **{conf_pct:.1f}%** — High")
                    else:
                        st.warning(f"🎯 Confidence: **{conf_pct:.1f}%** — Medium")

                st.markdown("#### 📊 All Category Probabilities")
                names = [p[0].capitalize() for p in all_probs]
                probs = [round(p[1] * 100, 1) for p in all_probs]
                fig2 = go.Figure(go.Bar(
                    x=probs, y=names, orientation="h",
                    marker_color=["#00D9FF" if (n.lower() == label.lower() and confidence >= 0.55) else "#1a3a32" for n in names],
                    text=[f"{p}%" for p in probs], textposition="outside",
                    textfont=dict(color="#E0F2F1")
                ))
                fig2.update_layout(
                    plot_bgcolor="#0d2b24", paper_bgcolor="#0d2b24",
                    font=dict(color="#E0F2F1", family="Inter"),
                    xaxis=dict(range=[0, 115], gridcolor="#1a3a32"),
                    yaxis=dict(gridcolor="#1a3a32"),
                    margin=dict(l=10, r=10, t=10, b=10), height=260
                )
                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("⬅️ Upload an image to get started.")

    if label and confidence >= 0.55:
        info = WASTE_INFO.get(label.lower(), {})
        score = SUSTAINABILITY_SCORE.get(label.lower(), 50)
        st.markdown("---")
        st.markdown(f'<div class="section-title">📋 {label.upper()} — Disposal Guide</div>', unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)
        g1.metric("🌱 Sustainability Score", f"{score}/100")
        g2.metric("⏱️ Decomposition Time", info.get("time", "N/A"))
        g3.metric("🎯 SDG Alignment", info.get("sdg", "N/A"))

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f'<div class="info-card"><h4>🗑️ Disposal Method</h4><p>{info.get("disposal","N/A")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card"><h4>💡 Sustainability Tips</h4><p>{info.get("recommendation","N/A")}</p></div>', unsafe_allow_html=True)
        with d2:
            st.markdown(f'<div class="info-card"><h4>🌍 Environmental Impact</h4><p>{info.get("impact","N/A")}</p></div>', unsafe_allow_html=True)
            score_pct = score
            st.markdown(f"""
            <div class="info-card">
                <h4>📈 Recyclability Score</h4>
                <p style="font-size:1.6rem;font-weight:900;color:#00D9FF">{score}/100</p>
                <div class="score-bar-wrap"><div class="score-bar" style="width:{score}%"></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🤖 IBM watsonx AI Sustainability Advisor")
        with st.spinner("Getting AI advice…"):
            try:
                ai_response = get_granite_response(
                    waste_type=label,
                    category=info.get("sdg", "General Waste"),
                    disposal=info.get("disposal", ""),
                    impact=info.get("impact", "")
                )
            except Exception as e:
                ai_response = f"AI service unavailable: {e}"
        st.success(ai_response)

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
elif mode == "📊 Dashboard":
    st.markdown('<div class="section-title">📊 EcoSort AI Dashboard</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Model Accuracy", "88.4%")
    m2.metric("📦 Waste Types", "6")
    m3.metric("🖼️ Training Images", "2,019")
    m4.metric("🔁 Epochs", "20")

    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("#### 📈 Model Performance")
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        scores_m = [88.4, 87.0, 86.0, 86.5]
        fig3 = go.Figure(go.Bar(
            x=scores_m, y=metrics, orientation="h",
            marker_color=["#00D9FF", "#00E5B3", "#4FC3F7", "#80CBC4"],
            text=[f"{s}%" for s in scores_m], textposition="outside",
            textfont=dict(color="#E0F2F1")
        ))
        fig3.update_layout(
            plot_bgcolor="#0d2b24", paper_bgcolor="#0d2b24",
            font=dict(color="#E0F2F1", family="Inter"),
            xaxis=dict(range=[0, 105], gridcolor="#1a3a32"),
            yaxis=dict(gridcolor="#1a3a32"),
            margin=dict(l=10, r=10, t=10, b=10), height=280
        )
        st.plotly_chart(fig3, use_container_width=True)

    with ch2:
        st.markdown("#### 🥧 Dataset Distribution")
        cats = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
        totals = [403, 501, 410, 594, 482, 137]
        fig4 = go.Figure(go.Pie(
            labels=cats, values=totals,
            hole=0.45,
            marker=dict(colors=[WASTE_INFO[c.lower()]["color"] for c in cats]),
            textfont=dict(color="#E0F2F1", size=12)
        ))
        fig4.update_layout(
            plot_bgcolor="#0d2b24", paper_bgcolor="#0d2b24",
            font=dict(color="#E0F2F1", family="Inter"),
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            legend=dict(font=dict(color="#E0F2F1"))
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🌱 Environmental Benefits")
    b1, b2, b3, b4 = st.columns(4)
    benefits = [
        ("🌿", "CO₂ Reduction", "2.5 tonnes CO₂ prevented per tonne sorted"),
        ("💧", "Water Saved", "Up to 7,000 gallons saved per recycled tonne"),
        ("⚡", "Energy Saved", "95% less energy with recycled aluminium"),
        ("🏭", "Landfill Relief", "Diverts waste from overflowing landfills"),
    ]
    for col, (ico, title, desc) in zip([b1, b2, b3, b4], benefits):
        col.markdown(f'<div class="metric-card"><span style="font-size:2rem">{ico}</span><span class="val" style="font-size:1rem;margin-top:8px">{title}</span><span class="lbl">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🗃️ Full Dataset Breakdown")
    df = pd.DataFrame({
        "Category": cats,
        "Training": [322, 400, 328, 475, 385, 109],
        "Validation": [81, 101, 82, 119, 97, 28],
        "Total": [403, 501, 410, 594, 482, 137]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── ABOUT ─────────────────────────────────────────────────────────────────────
elif mode == "ℹ️ About":
    st.markdown('<div class="section-title">ℹ️ About EcoSort AI</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        <div class="info-card">
            <h4>🤖 Model Architecture</h4>
            <p>YOLOv8 Nano Classification — 1.45M parameters, lightweight and fast. Trained on a balanced 6-class waste dataset.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h4>🔧 Tech Stack</h4>
            <p>YOLOv8 · IBM watsonx Granite · Streamlit · Plotly · Python · PIL</p>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="info-card">
            <h4>🌍 Mission</h4>
            <p>EcoSort AI empowers individuals to make sustainable waste management decisions through accessible, real-time AI classification.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h4>🎯 SDG Alignment</h4>
            <p>Aligned with UN SDG 12 (Responsible Consumption) and SDG 13 (Climate Action).</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Dataset Information")
    df_about = pd.DataFrame({
        "Category": ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"],
        "Training": [322, 400, 328, 475, 385, 109],
        "Validation": [81, 101, 82, 119, 97, 28],
        "Total": [403, 501, 410, 594, 482, 137]
    })
    st.dataframe(df_about, use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer-bar">♻️ EcoSort AI · Empowering Sustainable Waste Management Through AI · Powered by YOLOv8 + IBM watsonx</div>', unsafe_allow_html=True)