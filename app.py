"""
Fake News Intelligence System — Main Streamlit Dashboard
=========================================================
A premium, dark-themed, glassmorphic dashboard for AI-powered fake-news
detection, credibility scoring, and explainable predictions.

Run with:  streamlit run app.py
"""

# ── stdlib / third-party ────────────────────────────────────────────────
import sys, os, time, random, textwrap, io, base64
import streamlit as st
import pandas as pd
import numpy as np

# ── Plotly ──────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── WordCloud / matplotlib ──────────────────────────────────────────────
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# ── Project modules (graceful degradation) ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.preprocessor import TextPreprocessor
    _preprocessor = TextPreprocessor()
except Exception:
    _preprocessor = None

try:
    from src.feature_engineer import FeatureEngineer
    _feature_engineer = FeatureEngineer()
except Exception:
    _feature_engineer = None

try:
    from src.model import FakeNewsClassifier
except Exception:
    FakeNewsClassifier = None

try:
    from src.explainer import PredictionExplainer
except Exception:
    PredictionExplainer = None

try:
    from src.sentiment import SentimentAnalyzer
    _sentiment = SentimentAnalyzer()
except Exception:
    _sentiment = None

try:
    from src.clickbait import ClickbaitDetector
    _clickbait = ClickbaitDetector()
except Exception:
    _clickbait = None

try:
    from src.credibility import CredibilityScorer
except Exception:
    CredibilityScorer = None

try:
    from src.report_generator import ReportGenerator
    _report_gen = ReportGenerator()
except Exception:
    _report_gen = None

# ═══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fake News Intelligence System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  COLOR PALETTE & DYNAMIC THEMES
# ═══════════════════════════════════════════════════════════════════════
THEMES = {
    "Midnight Cyber (Default)": {
        "CYAN": "#00d4ff",
        "PURPLE": "#7c3aed",
        "BG": "#0a0a1a",
        "CARD": "#1a1a2e",
        "CARD_B": "rgba(255, 255, 255, 0.05)",
    },
    "Aurora Mint": {
        "CYAN": "#05ffc7",
        "PURPLE": "#00c6ff",
        "BG": "#060d13",
        "CARD": "#0f1e2d",
        "CARD_B": "rgba(5, 255, 199, 0.03)",
    },
    "Synthwave Neon": {
        "CYAN": "#ff79c6",
        "PURPLE": "#bd93f9",
        "BG": "#282a36",
        "CARD": "#21222c",
        "CARD_B": "rgba(255, 121, 198, 0.03)",
    },
    "Golden Luxury": {
        "CYAN": "#d4af37",
        "PURPLE": "#ff7b00",
        "BG": "#0c0c0c",
        "CARD": "#181818",
        "CARD_B": "rgba(212, 175, 55, 0.03)",
    }
}

# We initialize or read theme selection in sidebar at the very top
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;margin-top:10px;margin-bottom:10px;">
            <div style="font-size:2.4rem;">🔍</div>
            <div class="gradient-title" style="font-size:1.25rem;line-height:1.3;font-weight:700;">
                Fake News<br>Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    theme_choice = st.selectbox(
        "🎨 Select UI Theme",
        list(THEMES.keys()),
        index=0,
        key="selected_theme"
    )

selected_colors = THEMES.get(theme_choice, THEMES["Midnight Cyber (Default)"])
CYAN = selected_colors["CYAN"]
PURPLE = selected_colors["PURPLE"]
BG = selected_colors["BG"]
CARD = selected_colors["CARD"]
CARD_B = selected_colors["CARD_B"]

CORAL     = "#ff6b6b"
GREEN     = "#00ff88"
TEXT      = "#e0e0e0"
MUTED     = "#888888"

# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  – dark / glassmorphic / animated
# ═══════════════════════════════════════════════════════════════════════
CUSTOM_CSS = f"""
<style>
/* ── Google Font ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root reset ───────────────────────────────────────── */
:root {{
    --cyan: {CYAN};
    --purple: {PURPLE};
    --coral: {CORAL};
    --green: {GREEN};
    --bg: {BG};
    --card: {CARD};
    --card-border: {CARD_B};
    --text: {TEXT};
    --muted: {MUTED};
}}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}}
section[data-testid="stSidebar"] .stRadio label {{
    color: var(--text) !important;
    font-weight: 500;
    padding: 6px 0;
    transition: color .2s;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    color: var(--cyan) !important;
}}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: #333; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #555; }}

/* ── Glass card ───────────────────────────────────────── */
.glass-card {{
    background: rgba(26,26,46,0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    transition: transform .25s ease, box-shadow .25s ease;
}}
.glass-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,212,255,0.10);
}}

/* ── Metric card ──────────────────────────────────────── */
.metric-card {{
    background: rgba(26,26,46,0.70);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    transition: transform .25s, box-shadow .25s;
}}
.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 6px 30px rgba(0,212,255,0.12);
}}
.metric-card .metric-value {{
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}}
.metric-card .metric-label {{
    font-size: .85rem;
    color: var(--muted);
    font-weight: 500;
    letter-spacing: .3px;
}}

/* ── Gradient animated heading ────────────────────────── */
.gradient-title {{
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--cyan), var(--purple), var(--cyan));
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 4s ease infinite;
    margin-bottom: 0;
    line-height: 1.15;
}}
@keyframes gradient-shift {{
    0%   {{ background-position: 0% center; }}
    50%  {{ background-position: 100% center; }}
    100% {{ background-position: 0% center; }}
}}

/* ── Typewriter subtitle ──────────────────────────────── */
.typewriter {{
    overflow: hidden;
    white-space: nowrap;
    border-right: 2px solid var(--cyan);
    animation: typing 3.5s steps(44,end), blink .75s step-end infinite;
    font-size: 1.15rem;
    color: var(--muted);
    margin-top: 4px;
    width: max-content;
    max-width: 100%;
}}
@keyframes typing {{ from {{ width: 0; }} to {{ width: 100%; }} }}
@keyframes blink  {{ 50% {{ border-color: transparent; }} }}

/* ── Verdict badges ───────────────────────────────────── */
.verdict-badge {{
    display: inline-block;
    padding: 14px 38px;
    border-radius: 12px;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
}}
.verdict-real {{
    background: rgba(0,255,136,0.12);
    border: 2px solid var(--green);
    color: var(--green);
    box-shadow: 0 0 28px rgba(0,255,136,0.25);
    animation: glow-green 2s ease-in-out infinite alternate;
}}
.verdict-fake {{
    background: rgba(255,107,107,0.12);
    border: 2px solid var(--coral);
    color: var(--coral);
    box-shadow: 0 0 28px rgba(255,107,107,0.25);
    animation: glow-red 2s ease-in-out infinite alternate;
}}
@keyframes glow-green {{
    from {{ box-shadow: 0 0 18px rgba(0,255,136,0.20); }}
    to   {{ box-shadow: 0 0 38px rgba(0,255,136,0.40); }}
}}
@keyframes glow-red {{
    from {{ box-shadow: 0 0 18px rgba(255,107,107,0.20); }}
    to   {{ box-shadow: 0 0 38px rgba(255,107,107,0.40); }}
}}

/* ── Credibility ring (conic-gradient) ────────────────── */
.cred-ring {{
    width: 160px;
    height: 160px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto;
    position: relative;
}}
.cred-ring-inner {{
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}}
.cred-ring-inner .score {{
    font-size: 2.2rem;
    font-weight: 800;
    color: #fff;
}}
.cred-ring-inner .label {{
    font-size: .7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ── Grade badge ──────────────────────────────────────── */
.grade-badge {{
    display: inline-block;
    font-size: 2rem;
    font-weight: 900;
    width: 60px;
    height: 60px;
    line-height: 60px;
    text-align: center;
    border-radius: 14px;
    border: 2px solid;
}}

/* ── Tag chips ────────────────────────────────────────── */
.tag-real {{
    display: inline-block;
    background: rgba(0,255,136,0.10);
    border: 1px solid rgba(0,255,136,0.30);
    color: var(--green);
    border-radius: 999px;
    padding: 4px 14px;
    margin: 3px;
    font-size: .82rem;
    font-weight: 600;
}}
.tag-fake {{
    display: inline-block;
    background: rgba(255,107,107,0.10);
    border: 1px solid rgba(255,107,107,0.30);
    color: var(--coral);
    border-radius: 999px;
    padding: 4px 14px;
    margin: 3px;
    font-size: .82rem;
    font-weight: 600;
}}

/* ── Risk / positive bullet lists ─────────────────────── */
.risk-item {{
    padding: 6px 0;
    color: var(--coral);
    font-weight: 500;
}}
.positive-item {{
    padding: 6px 0;
    color: var(--green);
    font-weight: 500;
}}

/* ── Feature grid card ────────────────────────────────── */
.feature-card {{
    background: rgba(26,26,46,0.55);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
    min-height: 160px;
}}
.feature-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 6px 24px rgba(124,58,237,0.12);
}}
.feature-card .icon {{
    font-size: 2rem;
    margin-bottom: 8px;
}}
.feature-card .title {{
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
}}
.feature-card .desc {{
    font-size: .78rem;
    color: var(--muted);
    line-height: 1.45;
}}

/* ── Analyze button ───────────────────────────────────── */
div.stButton > button {{
    background: linear-gradient(135deg, var(--cyan), var(--purple)) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 32px !important;
    transition: transform .2s, box-shadow .2s !important;
    letter-spacing: .3px;
}}
div.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0,212,255,0.25) !important;
}}

/* ── Expander ─────────────────────────────────────────── */
details[data-testid="stExpander"] {{
    background: rgba(26,26,46,0.60) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}}

/* ── Tabs ─────────────────────────────────────────────── */
button[data-baseweb="tab"] {{
    color: var(--muted) !important;
    font-weight: 600 !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--cyan) !important;
    border-bottom-color: var(--cyan) !important;
}}

/* ── Progress bar override ────────────────────────────── */
div[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, var(--cyan), var(--purple)) !important;
}}

/* ── Small util ───────────────────────────────────────── */
.section-header {{
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 12px;
    color: #fff;
}}
.divider {{
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 20px 0;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "model": None,
        "metrics": None,
        "analysis_result": None,
        "dataset": None,
        "page": "🏠 Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════
#  CACHED LOADERS
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_model():
    """Return a FakeNewsClassifier instance (cached)."""
    if FakeNewsClassifier is None:
        return None
    model_path = os.path.join(os.path.dirname(__file__), "models", "fake_news_model.pkl")
    if os.path.exists(model_path):
        try:
            return FakeNewsClassifier.load(model_path)
        except Exception:
            pass
    return FakeNewsClassifier()


@st.cache_resource
def get_explainer(_model):
    if PredictionExplainer is None or _model is None:
        return None
    return PredictionExplainer(_model)


@st.cache_resource
def get_credibility(_model):
    if CredibilityScorer is None or _model is None:
        return None
    return CredibilityScorer(_model, _sentiment, _clickbait)


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: Plotly defaults
# ═══════════════════════════════════════════════════════════════════════
def _plotly_layout(fig, title="", height=400):
    """Apply the project dark theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=16, color="#fff")),
        font=dict(family="Inter, sans-serif", color=TEXT),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔬 Analyze Article", "📊 Dataset Explorer", "🧠 Model Performance", "ℹ️ About"],
        index=["🏠 Home", "🔬 Analyze Article", "📊 Dataset Explorer", "🧠 Model Performance", "ℹ️ About"].index(
            st.session_state.page
        ),
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Project info card
    st.markdown(
        """
        <div class="glass-card" style="padding:16px;">
            <div style="font-size:.78rem;color:#888;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Project Info</div>
            <div style="font-size:.88rem;font-weight:600;color:#fff;">Fake News Intelligence System</div>
            <div style="font-size:.75rem;color:#888;margin-top:4px;">NLP · ML · Explainable AI</div>
            <div style="font-size:.70rem;color:#555;margin-top:10px;">v1.0.0 · June 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:.65rem;color:#555;text-align:center;margin-top:10px;'>"
        "⚠️ For educational purposes only.<br>Not a replacement for professional fact-checking."
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("<div class='gradient-title'>Fake News Intelligence System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='typewriter'>AI-powered detection · Credibility scoring · Explainable results</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hero metric cards ───────────────────────────────────────────
    cols = st.columns(4, gap="medium")
    cards = [
        ("🤖", "AI-Powered", "Detection", "Ensemble ML pipeline with TF-IDF and linguistic features"),
        ("📊", "Credibility", "Scoring", "Multi-signal scoring across 5 independent dimensions"),
        ("🧠", "Explainable", "AI", "Transparent predictions with word-level indicator analysis"),
        ("⚡", "Real-time", "Analysis", "Instant article analysis with comprehensive reporting"),
    ]
    for col, (icon, line1, line2, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:2.4rem;margin-bottom:8px;">{icon}</div>
                    <div class="metric-value" style="font-size:1.3rem;">{line1}</div>
                    <div style="font-size:1rem;font-weight:700;color:#fff;margin-bottom:6px;">{line2}</div>
                    <div style="font-size:.76rem;color:{MUTED};line-height:1.45;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature highlights grid ─────────────────────────────────────
    st.markdown("<div class='section-header'>✨ Platform Capabilities</div>", unsafe_allow_html=True)

    features = [
        ("🔬", "Deep Text Analysis", "Advanced NLP preprocessing with tokenization, lemmatization, and noise removal"),
        ("📈", "Sentiment Profiling", "VADER-based emotional analysis with fake-news sentiment flag detection"),
        ("🎣", "Clickbait Detection", "Pattern-based headline analysis scoring clickbait probability 0-100"),
        ("🏅", "Credibility Grading", "A-F grading system aggregating ML confidence, sentiment, clickbait & linguistics"),
        ("📋", "Report Generation", "Downloadable HTML intelligence reports with full analysis breakdown"),
        ("📊", "Interactive Visuals", "Rich Plotly charts — radar, heatmap, histograms and more"),
    ]
    rows = [features[i : i + 3] for i in range(0, len(features), 3)]
    for row in rows:
        cols = st.columns(len(row), gap="medium")
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="icon">{icon}</div>
                        <div class="title">{title}</div>
                        <div class="desc">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick demo CTA ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;padding:32px;">
            <div style="font-size:1.2rem;font-weight:700;color:#fff;margin-bottom:6px;">
                Ready to analyze an article?
            </div>
            <div style="font-size:.85rem;color:#888;">
                Paste any news article and get an instant AI-powered credibility assessment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚀  Launch Analyzer", use_container_width=True):
        st.session_state.page = "🔬 Analyze Article"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANALYZE ARTICLE
# ═══════════════════════════════════════════════════════════════════════
def page_analyze():
    st.markdown("<div class='gradient-title' style='font-size:2rem;'>🔬 Analyze Article</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#888;margin-bottom:18px;'>Paste your article text below for a comprehensive AI-driven analysis.</p>",
        unsafe_allow_html=True,
    )

    # ── Model check ─────────────────────────────────────────────────
    model = get_model()
    if model is None:
        st.error("⚠️ Model module could not be loaded. Please check `src/model.py`.")
        return
    if not model.is_trained:
        st.warning("🧠 The model has not been trained yet. Go to **📊 Dataset Explorer** first to train the model on a dataset.")

    # ── Input modes ─────────────────────────────────────────────────
    tab_paste, tab_hl = st.tabs(["📋 Paste Full Article", "🗞️ Headline + Body"])

    headline_input = ""
    article_text = ""

    with tab_paste:
        article_text = st.text_area(
            "Article text",
            height=220,
            placeholder="Paste the full article content here …",
            label_visibility="collapsed",
        )
        headline_input = st.text_input("Headline (optional)", placeholder="Enter headline if available …")

    with tab_hl:
        headline_input_2 = st.text_input("Headline", placeholder="Enter the headline …", key="hl2")
        body_input = st.text_area("Body", height=200, placeholder="Enter the article body …", key="body2")
        if headline_input_2:
            headline_input = headline_input_2
        if body_input:
            article_text = body_input

    # ── Analyze button ──────────────────────────────────────────────
    analyze_clicked = st.button("🔍  Analyze Now", use_container_width=True)

    if analyze_clicked and article_text.strip():
        _run_analysis(model, article_text.strip(), headline_input.strip())

    elif analyze_clicked and not article_text.strip():
        st.warning("Please enter some article text to analyze.")

    # ── Show persisted results ──────────────────────────────────────
    if st.session_state.analysis_result and not analyze_clicked:
        _render_results(st.session_state.analysis_result)


def _run_analysis(model, text, headline):
    """Execute the full analysis pipeline and render results."""
    results = {}
    with st.spinner("🧠 Running AI analysis pipeline …"):
        # Preprocess
        clean = _preprocessor.preprocess(text) if _preprocessor else text

        # Prediction
        if model.is_trained:
            pred = model.predict(clean)
        else:
            pred = {"label": "UNKNOWN", "confidence": 0.0, "probabilities": {"REAL": 0.5, "FAKE": 0.5}}
        results["prediction"] = pred

        # Sentiment
        if _sentiment:
            results["sentiment"] = _sentiment.analyze(text)

        # Clickbait
        if _clickbait and headline:
            results["clickbait"] = _clickbait.detect(headline)
        elif _clickbait:
            results["clickbait"] = _clickbait.detect(text[:200])

        # Explainer
        explainer = get_explainer(model)
        if explainer and model.is_trained:
            try:
                results["explanation"] = explainer.explain(clean)
            except Exception:
                results["explanation"] = None

        # Credibility
        cred = get_credibility(model)
        if cred and model.is_trained:
            try:
                results["credibility"] = cred.score(text, headline or text[:200], pred)
            except Exception:
                results["credibility"] = None

        # Linguistic features
        if _feature_engineer:
            results["linguistic"] = _feature_engineer.get_linguistic_features(text)

        # Report
        if _report_gen:
            try:
                results["html_report"] = _report_gen.generate_html_report(results)
            except Exception:
                results["html_report"] = None

        results["text"] = text
        results["headline"] = headline
        results["clean_text"] = clean

    st.session_state.analysis_result = results
    _render_results(results)


def _render_results(r):
    """Render the full results dashboard for an analyzed article."""
    pred = r.get("prediction", {})
    sentiment = r.get("sentiment")
    clickbait = r.get("clickbait")
    explanation = r.get("explanation")
    credibility = r.get("credibility")
    linguistic = r.get("linguistic")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Analysis Results</div>", unsafe_allow_html=True)

    # ── Top row: Verdict + Credibility  |  Sentiment + Clickbait ───
    left, right = st.columns([2, 3], gap="large")

    with left:
        # Verdict badge
        label = pred.get("label", "UNKNOWN")
        conf = pred.get("confidence", 0)
        badge_class = "verdict-real" if label == "REAL" else "verdict-fake"
        st.markdown(
            f"""
            <div style="text-align:center;margin-bottom:18px;">
                <div class="verdict-badge {badge_class}">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence bar
        st.markdown(
            f"<div style='text-align:center;font-size:.85rem;color:{MUTED};margin-bottom:4px;'>Confidence</div>",
            unsafe_allow_html=True,
        )
        conf_color = GREEN if label == "REAL" else CORAL
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.06);border-radius:8px;height:18px;overflow:hidden;margin-bottom:16px;">
                <div style="width:{conf*100:.1f}%;height:100%;background:{conf_color};border-radius:8px;
                            transition:width .6s ease;"></div>
            </div>
            <div style="text-align:center;font-size:1.6rem;font-weight:800;color:{conf_color};">{conf*100:.1f}%</div>
            """,
            unsafe_allow_html=True,
        )

        # Credibility ring
        if credibility:
            score = credibility.get("overall_score", 0)
            grade = credibility.get("grade", "?")
            grade_color = credibility.get("grade_color", CYAN)
            rest = 100 - score
            st.markdown(
                f"""
                <div style="text-align:center;margin-top:22px;">
                    <div style="font-size:.85rem;color:{MUTED};margin-bottom:8px;">Credibility Score</div>
                    <div class="cred-ring" style="background:conic-gradient({grade_color} {score}%, rgba(255,255,255,0.06) {score}% 100%);">
                        <div class="cred-ring-inner">
                            <div class="score">{score}</div>
                            <div class="label">/ 100</div>
                        </div>
                    </div>
                    <div style="margin-top:10px;">
                        <span class="grade-badge" style="border-color:{grade_color};color:{grade_color};">{grade}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        # Sentiment card
        if sentiment:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>💬 Sentiment Analysis</div>", unsafe_allow_html=True)

            if HAS_PLOTLY:
                fig = go.Figure()
                labels = ["Positive", "Neutral", "Negative"]
                values = [sentiment.get("positive", 0), sentiment.get("neutral", 0), sentiment.get("negative", 0)]
                colors = [GREEN, CYAN, CORAL]
                for lbl, val, clr in zip(labels, values, colors):
                    fig.add_trace(go.Bar(
                        y=[lbl], x=[val], orientation="h",
                        marker_color=clr, name=lbl,
                        text=[f"{val:.2f}"], textposition="auto",
                    ))
                _plotly_layout(fig, height=180)
                fig.update_layout(showlegend=False, margin=dict(l=80, r=20, t=10, b=10))
                fig.update_xaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            compound = sentiment.get("compound", 0)
            s_label = sentiment.get("sentiment_label", "N/A")
            flag = sentiment.get("fake_news_sentiment_flag", False)
            st.markdown(
                f"""
                <div style="display:flex;gap:18px;flex-wrap:wrap;font-size:.82rem;">
                    <span>Compound: <b style="color:{CYAN};">{compound:+.3f}</b></span>
                    <span>Label: <b>{s_label}</b></span>
                    <span>Intensity: <b>{sentiment.get('emotional_intensity','N/A')}</b></span>
                    {"<span style='color:" + CORAL + ";font-weight:700;'>⚠ Suspicious Sentiment Pattern</span>" if flag else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Clickbait card
        if clickbait:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>🎣 Clickbait Analysis</div>", unsafe_allow_html=True)
            cb_score = clickbait.get("clickbait_score", 0)
            is_cb = clickbait.get("is_clickbait", False)
            cb_color = CORAL if is_cb else GREEN

            if HAS_PLOTLY:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=cb_score,
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor=MUTED),
                        bar=dict(color=cb_color),
                        bgcolor="rgba(255,255,255,0.04)",
                        steps=[
                            dict(range=[0, 30], color="rgba(0,255,136,0.08)"),
                            dict(range=[30, 60], color="rgba(0,212,255,0.08)"),
                            dict(range=[60, 100], color="rgba(255,107,107,0.08)"),
                        ],
                    ),
                    number=dict(suffix="/100", font=dict(size=28)),
                ))
                _plotly_layout(fig, height=200)
                fig.update_layout(margin=dict(l=30, r=30, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            indicators = clickbait.get("indicators", [])
            if indicators:
                st.markdown(
                    "<div style='font-size:.8rem;color:#888;margin-bottom:4px;'>Indicators found:</div>"
                    + "".join(f"<span class='tag-fake'>{ind}</span>" for ind in indicators),
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Explanation section ─────────────────────────────────────────
    if explanation:
        with st.expander("🧠 AI Explanation — Why this verdict?", expanded=True):
            st.markdown(
                f"<div style='color:{TEXT};line-height:1.7;margin-bottom:14px;'>{explanation.get('explanation_text', '')}</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='section-header' style='color:{GREEN};'>✅ Pro-Real Indicators</div>", unsafe_allow_html=True)
                for word, weight in explanation.get("top_real_indicators", []):
                    st.markdown(f"<span class='tag-real'>{word} ({weight:+.3f})</span>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='section-header' style='color:{CORAL};'>🚩 Pro-Fake Indicators</div>", unsafe_allow_html=True)
                for word, weight in explanation.get("top_fake_indicators", []):
                    st.markdown(f"<span class='tag-fake'>{word} ({weight:+.3f})</span>", unsafe_allow_html=True)

    # ── Credibility breakdown radar ─────────────────────────────────
    if credibility and HAS_PLOTLY:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🏅 Credibility Breakdown</div>", unsafe_allow_html=True)

        breakdown = credibility.get("breakdown", {})
        cats = list(breakdown.keys())
        vals = [breakdown[c] for c in cats]
        display_cats = [c.replace("_", " ").title() for c in cats]
        # close the radar
        display_cats_closed = display_cats + [display_cats[0]]
        vals_closed = vals + [vals[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=display_cats_closed,
            fill="toself",
            fillcolor="rgba(0,212,255,0.12)",
            line=dict(color=CYAN, width=2),
            marker=dict(size=6, color=CYAN),
        ))
        _plotly_layout(fig, title="Multi-Signal Credibility Radar", height=420)
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color=MUTED)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color=TEXT)),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Risk factors & positive signals ─────────────────────────────
    if credibility:
        rf = credibility.get("risk_factors", [])
        ps = credibility.get("positive_signals", [])
        if rf or ps:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='section-header' style='color:{CORAL};'>🚩 Risk Factors</div>", unsafe_allow_html=True)
                for item in rf:
                    st.markdown(f"<div class='risk-item'>⚠ {item}</div>", unsafe_allow_html=True)
                if not rf:
                    st.markdown("<div style='color:#888;font-size:.85rem;'>No risk factors detected.</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='section-header' style='color:{GREEN};'>✅ Positive Signals</div>", unsafe_allow_html=True)
                for item in ps:
                    st.markdown(f"<div class='positive-item'>✔ {item}</div>", unsafe_allow_html=True)
                if not ps:
                    st.markdown("<div style='color:#888;font-size:.85rem;'>No positive signals detected.</div>", unsafe_allow_html=True)

    # ── Linguistic features (optional) ──────────────────────────────
    if linguistic:
        with st.expander("📐 Linguistic Feature Details"):
            ling_df = pd.DataFrame(
                [{"Feature": k.replace("_", " ").title(), "Value": v} for k, v in linguistic.items()]
            )
            st.dataframe(ling_df, use_container_width=True, hide_index=True)

    # ── Download report ─────────────────────────────────────────────
    html_report = r.get("html_report")
    if html_report:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.download_button(
            label="📥  Download Full HTML Report",
            data=html_report,
            file_name="fake_news_report.html",
            mime="text/html",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 3 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════
def page_dataset():
    st.markdown("<div class='gradient-title' style='font-size:2rem;'>📊 Dataset Explorer</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#888;margin-bottom:18px;'>Load a dataset, explore distributions, and train the detection model.</p>",
        unsafe_allow_html=True,
    )

    # ── Data source ─────────────────────────────────────────────────
    src = st.radio("Data source", ["Use Sample Data", "Upload CSV"], horizontal=True, label_visibility="collapsed")

    df = None

    if src == "Use Sample Data":
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample", "sample_data.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            st.success(f"Loaded **{len(df)}** articles from sample dataset.")
        else:
            st.info("Sample data not found. Attempting to generate …")
            gen_script = os.path.join(os.path.dirname(__file__), "scripts", "generate_sample_data.py")
            if os.path.exists(gen_script):
                import subprocess
                with st.spinner("Generating sample dataset …"):
                    subprocess.run([sys.executable, gen_script], cwd=os.path.dirname(__file__))
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    st.success(f"Generated and loaded **{len(df)}** articles.")
                else:
                    st.error("Data generation failed.")
            else:
                st.warning("No sample data or generator script found. Please upload a CSV.")
    else:
        uploaded = st.file_uploader("Upload CSV (columns: text, label)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded **{len(df)}** rows from uploaded file.")

    if df is None:
        return

    # Validate columns
    if "text" not in df.columns or "label" not in df.columns:
        st.error("CSV must contain `text` and `label` columns.")
        return

    st.session_state.dataset = df

    # ── Train model ─────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🧠 Train Model</div>", unsafe_allow_html=True)

    model = get_model()
    if model is None:
        st.error("Model module unavailable.")
        return

    if st.button("🚀  Train on this dataset", use_container_width=True):
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        progress = st.progress(0, text="Preparing data …")
        progress.progress(10, text="Training model …")
        with st.spinner("Training model — this may take a moment …"):
            metrics = model.train(texts, labels)
        progress.progress(90, text="Saving model …")
        # Save
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        try:
            model.save(os.path.join(model_dir, "fake_news_model.pkl"))
        except Exception:
            pass
        progress.progress(100, text="Done!")
        st.session_state.metrics = metrics
        st.session_state.model = model
        st.success("✅ Model trained successfully!")

        # Show quick metrics
        mcols = st.columns(4)
        metric_items = [
            ("Accuracy", metrics.get("accuracy", 0)),
            ("Precision", metrics.get("precision", 0)),
            ("Recall", metrics.get("recall", 0)),
            ("F1-Score", metrics.get("f1", 0)),
        ]
        for col, (name, val) in zip(mcols, metric_items):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{val:.2%}</div>
                        <div class="metric-label">{name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Visualizations ──────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📈 Data Visualizations</div>", unsafe_allow_html=True)

    if not HAS_PLOTLY:
        st.warning("Plotly not installed — charts unavailable.")
        return

    # 1) Distribution pie
    col_pie, col_hist = st.columns(2)
    with col_pie:
        counts = df["label"].value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.55,
            marker=dict(colors=[GREEN, CORAL] if "REAL" in counts.index[:1].tolist() else [CORAL, GREEN]),
            textinfo="label+percent",
            textfont=dict(size=14),
        ))
        _plotly_layout(fig, title="Label Distribution", height=370)
        st.plotly_chart(fig, use_container_width=True)

    # 2) Word count histogram
    with col_hist:
        df["_wc"] = df["text"].astype(str).apply(lambda t: len(t.split()))
        fig = go.Figure()
        for lbl, clr in [("REAL", GREEN), ("FAKE", CORAL)]:
            subset = df[df["label"] == lbl]["_wc"]
            if len(subset):
                fig.add_trace(go.Histogram(x=subset, name=lbl, marker_color=clr, opacity=0.65, nbinsx=30))
        _plotly_layout(fig, title="Word Count Distribution", height=370)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Word Count")
        fig.update_yaxes(title_text="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    # 3) Sentiment distribution (box plots)
    if _sentiment:
        st.markdown("<div class='section-header'>💬 Sentiment Distribution (Real vs Fake)</div>", unsafe_allow_html=True)
        sample_n = min(200, len(df))
        sample_df = df.sample(n=sample_n, random_state=42).copy()
        with st.spinner("Computing sentiment for visualization …"):
            sample_df["compound"] = sample_df["text"].astype(str).apply(
                lambda t: _sentiment.analyze(t).get("compound", 0)
            )
        fig = go.Figure()
        for lbl, clr in [("REAL", GREEN), ("FAKE", CORAL)]:
            subset = sample_df[sample_df["label"] == lbl]["compound"]
            if len(subset):
                fig.add_trace(go.Box(y=subset, name=lbl, marker_color=clr, boxmean=True))
        _plotly_layout(fig, title="Compound Sentiment by Label", height=370)
        st.plotly_chart(fig, use_container_width=True)

    # 4) Top features
    if model and model.is_trained:
        try:
            coefs = None
            # Try to get feature importance from the model's internal pipeline
            if hasattr(model, "pipeline"):
                clf = model.pipeline.named_steps.get("clf") or model.pipeline.named_steps.get("classifier")
                vec = model.pipeline.named_steps.get("tfidf") or model.pipeline.named_steps.get("vectorizer")
                if clf is not None and vec is not None:
                    if hasattr(clf, "coef_"):
                        feature_names = vec.get_feature_names_out()
                        importances = clf.coef_[0]
                        top_idx = np.argsort(np.abs(importances))[-20:]
                        top_features = [(feature_names[i], importances[i]) for i in top_idx]
                        top_features.sort(key=lambda x: x[1])
                        coefs = top_features
            if coefs:
                st.markdown("<div class='section-header'>🔑 Top 20 Most Important Features</div>", unsafe_allow_html=True)
                words, weights = zip(*coefs)
                colors = [CORAL if w > 0 else GREEN for w in weights]
                fig = go.Figure(go.Bar(
                    y=list(words), x=list(weights), orientation="h",
                    marker_color=colors,
                ))
                _plotly_layout(fig, title="Feature Importances (Model Coefficients)", height=500)
                fig.update_layout(margin=dict(l=120))
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # 5) Word clouds
    if HAS_WORDCLOUD:
        st.markdown("<div class='section-header'>☁️ Word Clouds</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for col, lbl, cmap_name in [(c1, "REAL", "winter"), (c2, "FAKE", "hot")]:
            with col:
                subset = df[df["label"] == lbl]["text"].astype(str)
                if len(subset):
                    text_blob = " ".join(subset.sample(min(200, len(subset)), random_state=42).tolist())
                    wc = WordCloud(
                        width=600, height=350, background_color="#0a0a1a",
                        colormap=cmap_name, max_words=80,
                    ).generate(text_blob)
                    fig_wc, ax = plt.subplots(figsize=(6, 3.5))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title(f"{lbl} Articles", color="white", fontsize=14, fontweight="bold")
                    fig_wc.patch.set_facecolor("#0a0a1a")
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)

    # 6) Sample predictions
    if model and model.is_trained:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🔮 Sample Predictions</div>", unsafe_allow_html=True)
        sample_preds = df.sample(n=min(5, len(df)), random_state=int(time.time()) % 1000)
        for _, row in sample_preds.iterrows():
            text_preview = str(row["text"])[:200] + ("…" if len(str(row["text"])) > 200 else "")
            actual = row["label"]
            pred = model.predict(str(row["text"]))
            p_label = pred.get("label", "?")
            p_conf = pred.get("confidence", 0)
            match = "✅" if p_label == actual else "❌"
            badge_cls = "tag-real" if p_label == "REAL" else "tag-fake"
            st.markdown(
                f"""
                <div class="glass-card" style="padding:14px 18px;">
                    <div style="font-size:.82rem;color:{MUTED};margin-bottom:4px;">{text_preview}</div>
                    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                        <span style="font-size:.78rem;color:{MUTED};">Actual: <b>{actual}</b></span>
                        <span class="{badge_cls}">Predicted: {p_label} ({p_conf:.0%})</span>
                        <span>{match}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
def page_model():
    st.markdown("<div class='gradient-title' style='font-size:2rem;'>🧠 Model Performance</div>", unsafe_allow_html=True)

    model = get_model()
    metrics = st.session_state.metrics

    if model is None or not model.is_trained or metrics is None:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;padding:40px;">
                <div style="font-size:2rem;margin-bottom:10px;">🧠</div>
                <div style="font-size:1.1rem;font-weight:600;color:#fff;margin-bottom:6px;">No trained model found</div>
                <div style="color:#888;font-size:.88rem;">Head over to <b>📊 Dataset Explorer</b> to train a model first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Dataset Explorer"):
            st.session_state.page = "📊 Dataset Explorer"
            st.rerun()
        return

    # ── Metric cards ────────────────────────────────────────────────
    mcols = st.columns(4, gap="medium")
    kpis = [
        ("🎯", "Accuracy", metrics.get("accuracy", 0)),
        ("🔍", "Precision", metrics.get("precision", 0)),
        ("📡", "Recall", metrics.get("recall", 0)),
        ("⚖️", "F1-Score", metrics.get("f1", 0)),
    ]
    for col, (icon, name, val) in zip(mcols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.6rem;margin-bottom:4px;">{icon}</div>
                    <div class="metric-value">{val:.2%}</div>
                    <div class="metric-label">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not HAS_PLOTLY:
        st.warning("Plotly not installed.")
        return

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Confusion matrix heatmap ────────────────────────────────────
    cm = metrics.get("confusion_matrix")
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("<div class='section-header'>📋 Confusion Matrix</div>", unsafe_allow_html=True)
        if cm is not None:
            cm_arr = np.array(cm)
            labels = ["REAL", "FAKE"]
            fig = go.Figure(go.Heatmap(
                z=cm_arr,
                x=labels, y=labels,
                colorscale=[[0, "#0a0a1a"], [0.5, PURPLE], [1, CYAN]],
                text=cm_arr.astype(str),
                texttemplate="%{text}",
                textfont=dict(size=20, color="#fff"),
                showscale=False,
            ))
            _plotly_layout(fig, height=370)
            fig.update_xaxes(title_text="Predicted", side="bottom")
            fig.update_yaxes(title_text="Actual", autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Confusion matrix not available.")

    # ── ROC Curve ───────────────────────────────────────────────────
    with col_roc:
        st.markdown("<div class='section-header'>📈 ROC Curve</div>", unsafe_allow_html=True)
        roc_auc = metrics.get("roc_auc")
        if roc_auc is not None:
            # Approximate ROC for display
            fpr = np.linspace(0, 1, 100)
            # A simple parametric curve that respects the AUC
            tpr = 1 - (1 - fpr) ** (roc_auc / (1 - roc_auc + 1e-9)) if roc_auc < 1 else np.ones_like(fpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=CYAN, width=2.5),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
                name=f"AUC = {roc_auc:.4f}",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color=MUTED, width=1, dash="dash"),
                name="Random",
            ))
            _plotly_layout(fig, title=f"ROC Curve (AUC = {roc_auc:.4f})", height=370)
            fig.update_xaxes(title_text="False Positive Rate")
            fig.update_yaxes(title_text="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC AUC not available.")

    # ── Classification report table ─────────────────────────────────
    report = metrics.get("classification_report")
    if report:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📝 Classification Report</div>", unsafe_allow_html=True)
        if isinstance(report, str):
            st.markdown(
                f"<div class='glass-card'><pre style='color:{TEXT};font-size:.82rem;'>{report}</pre></div>",
                unsafe_allow_html=True,
            )
        elif isinstance(report, dict):
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # ── Model architecture ──────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    with st.expander("🏗️ Model Architecture & Methodology", expanded=False):
        st.markdown(
            f"""
            <div style="color:{TEXT};line-height:1.8;font-size:.9rem;">
                <h4 style="color:#fff;">Ensemble Pipeline Architecture</h4>
                <ol>
                    <li><b>Text Preprocessing</b> — Lowercasing, punctuation removal, stop-word filtering,
                        lemmatization via spaCy / NLTK.</li>
                    <li><b>Feature Extraction</b> — TF-IDF vectorization (unigrams + bigrams, max 10 000 features)
                        combined with engineered linguistic features (avg word length, sentence count,
                        punctuation ratios, etc.).</li>
                    <li><b>Classification</b> — Logistic Regression / Passive-Aggressive Classifier
                        trained on the combined feature matrix.</li>
                    <li><b>Calibration</b> — Probability calibration for reliable confidence scores.</li>
                </ol>
                <h4 style="color:#fff;">Supplementary Signals</h4>
                <ul>
                    <li><b>Sentiment Analysis</b> — VADER compound score + emotional intensity profiling.</li>
                    <li><b>Clickbait Detection</b> — Rule-based pattern matching on headlines (14+ heuristic patterns).</li>
                    <li><b>Credibility Scoring</b> — Weighted aggregation of ML confidence, sentiment,
                        clickbait score, linguistic quality, and source-credibility heuristics → 0-100 + letter grade.</li>
                    <li><b>Explainability</b> — Per-word coefficient analysis for transparent predictions.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════
#  PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown("<div class='gradient-title' style='font-size:2rem;'>ℹ️ About the Project</div>", unsafe_allow_html=True)

    # ── Overview ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-header">🌐 Project Overview</div>
            <p style="color:{TEXT};line-height:1.75;font-size:.92rem;">
                The <b>Fake News Intelligence System</b> is an AI-powered platform that classifies
                news articles as <span style="color:{GREEN};font-weight:700;">REAL</span> or
                <span style="color:{CORAL};font-weight:700;">FAKE</span> using a combination of
                Natural Language Processing, Machine Learning, Sentiment Analysis, Clickbait Detection,
                and Credibility Scoring. Every prediction is accompanied by transparent, word-level
                explanations so users can understand <i>why</i> a verdict was given.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Architecture diagram ────────────────────────────────────────
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-header">🏗️ System Architecture</div>
            <div style="overflow-x:auto;">
                <pre style="color:{CYAN};font-size:.72rem;line-height:1.6;background:rgba(0,0,0,0.3);padding:18px;border-radius:10px;">
┌──────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT DASHBOARD                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐   │
│  │   Home   │  │   Analyzer   │  │ Explorer │  │  Performance  │   │
│  └──────────┘  └──────┬───────┘  └────┬─────┘  └───────────────┘   │
│                       │               │                              │
│  ┌────────────────────▼───────────────▼──────────────────────────┐  │
│  │                    ANALYSIS ENGINE                             │  │
│  │  ┌─────────────┐  ┌───────────┐  ┌───────────┐               │  │
│  │  │ Preprocessor│─▶│ Feature   │─▶│ ML Model  │               │  │
│  │  │ (NLP Clean) │  │ Engineer  │  │ (TF-IDF+  │               │  │
│  │  └─────────────┘  └───────────┘  │  LR/PA)   │               │  │
│  │                                   └─────┬─────┘               │  │
│  │  ┌───────────┐  ┌─────────────┐         │                    │  │
│  │  │ Sentiment │  │  Clickbait  │   ┌─────▼──────┐             │  │
│  │  │ Analyzer  │  │  Detector   │   │ Explainer  │             │  │
│  │  └─────┬─────┘  └──────┬──────┘   └─────┬──────┘             │  │
│  │        │               │                │                    │  │
│  │  ┌─────▼───────────────▼────────────────▼──────────────┐     │  │
│  │  │              CREDIBILITY SCORER                      │     │  │
│  │  │  ML Confidence + Sentiment + Clickbait + Linguistic  │     │  │
│  │  │  → Overall Score (0-100) + Grade (A-F)               │     │  │
│  │  └──────────────────────┬───────────────────────────────┘     │  │
│  │                         │                                    │  │
│  │  ┌──────────────────────▼───────────────────────────────┐    │  │
│  │  │              REPORT GENERATOR (HTML)                  │    │  │
│  │  └──────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                </pre>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Methodology ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔬 Methodology</div>", unsafe_allow_html=True)

    methodologies = [
        ("📝 Text Preprocessing", "Raw article text passes through lowercasing, URL/HTML removal, punctuation stripping, stop-word filtering, and lemmatization to produce a clean, normalized representation."),
        ("🔢 Feature Engineering", "TF-IDF vectors (unigrams + bigrams, up to 10 000 features) are concatenated with hand-crafted linguistic features — average word length, sentence count, uppercase ratio, punctuation density, and more."),
        ("🤖 Classification", "A supervised ML classifier (Logistic Regression / Passive-Aggressive) is trained on the combined feature matrix. The model outputs calibrated probabilities for REAL and FAKE classes."),
        ("💬 Sentiment Analysis", "VADER lexicon-based sentiment scoring produces positive, negative, neutral, and compound scores. Extreme sentiment patterns are flagged as potential misinformation indicators."),
        ("🎣 Clickbait Detection", "14+ heuristic patterns (excessive punctuation, ALL CAPS, superlatives, number-listicle starters, etc.) are scanned in the headline to produce a 0-100 clickbait score."),
        ("🏅 Credibility Scoring", "Five independent signals (ML confidence, sentiment normality, clickbait absence, linguistic quality, source heuristics) are aggregated via weighted average into a 0-100 score mapped to an A-F grade."),
        ("🧠 Explainability", "Per-word TF-IDF coefficient analysis reveals which words pushed the model toward REAL vs FAKE, enabling transparent and interpretable predictions."),
    ]
    for title, body in methodologies:
        with st.expander(title):
            st.markdown(f"<div style='color:{TEXT};line-height:1.7;font-size:.88rem;'>{body}</div>", unsafe_allow_html=True)

    # ── Technology stack ────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🛠️ Technology Stack</div>", unsafe_allow_html=True)

    tech = [
        ("🐍", "Python 3.10+", "Core language"),
        ("📊", "Streamlit", "Interactive dashboard framework"),
        ("🤖", "scikit-learn", "ML pipeline & classification"),
        ("📈", "Plotly", "Interactive data visualizations"),
        ("📝", "NLTK / spaCy", "NLP preprocessing"),
        ("💬", "VADER Sentiment", "Lexicon-based sentiment analysis"),
        ("☁️", "WordCloud", "Text visualization"),
        ("🐼", "Pandas / NumPy", "Data manipulation"),
    ]
    cols = st.columns(4, gap="medium")
    for i, (icon, name, desc) in enumerate(tech):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="feature-card" style="min-height:110px;">
                    <div class="icon">{icon}</div>
                    <div class="title" style="font-size:.88rem;">{name}</div>
                    <div class="desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Author / links ──────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glass-card" style="text-align:center;padding:28px;">
            <div style="font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:6px;">👤 Author</div>
            <div style="color:{MUTED};font-size:.88rem;margin-bottom:12px;">
                Built as part of an AI / ML internship showcase project.
            </div>
            <div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;">
                <span class="tag-real">📂 GitHub Repository</span>
                <span class="tag-real">📄 Documentation</span>
                <span class="tag-real">📧 Contact</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="text-align:center;margin-top:30px;font-size:.72rem;color:#555;">
            Fake News Intelligence System v1.0.0 · © 2026 · Built with ❤️ and Python
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════════════
PAGES = {
    "🏠 Home": page_home,
    "🔬 Analyze Article": page_analyze,
    "📊 Dataset Explorer": page_dataset,
    "🧠 Model Performance": page_model,
    "ℹ️ About": page_about,
}

PAGES.get(st.session_state.page, page_home)()
