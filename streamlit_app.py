"""
PondIQ Streamlit App — Responsive feeding advisor for fish farmers.

Run:
    streamlit run streamlit_app.py

Requires the PondIQ FastAPI server running at http://localhost:8000
    cd PondIQ-main && uv run pondiq_api.py
"""

from __future__ import annotations

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="PondIQ — Fish Feeding Advisor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Responsive CSS ───────────────────────────────────────────────────


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Base ─────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700&family=Roboto+Slab:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Noto Sans', sans-serif;
        }

        h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2 {
            font-family: 'Roboto Slab', serif !important;
            color: #0f2340;
        }

        /* ── Global background ────────────────────────────── */
        .stApp {
            background: #eef4fb;
        }

        /* ── Max-width container for mobile-friendly feel ─── */
        .block-container {
            max-width: 720px !important;
            padding: 1rem 1rem 2rem !important;
        }

        /* ── Cards ────────────────────────────────────────── */
        .pondiq-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(15,35,64,0.09);
            box-shadow: 0 2px 10px rgba(15,35,64,0.06);
            margin-bottom: 14px;
        }

        /* ── Decision banners ─────────────────────────────── */
        .decision-banner {
            border-radius: 16px;
            padding: 22px;
            position: relative;
            overflow: hidden;
            color: #ffffff;
            margin-bottom: 14px;
        }
        .decision-banner.feed-now   { background: #0e7a3e; }
        .decision-banner.reduce     { background: #b45309; }
        .decision-banner.stop       { background: #b91c1c; }

        .decision-banner .label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            opacity: 0.7;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .decision-banner .title {
            font-family: 'Roboto Slab', serif;
            font-size: 24px;
            font-weight: 700;
        }

        /* ── Parameter cards ──────────────────────────────── */
        .param-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 16px;
            border: 1px solid rgba(15,35,64,0.09);
            box-shadow: 0 1px 6px rgba(15,35,64,0.06);
            margin-bottom: 10px;
        }

        /* ── Tip row ──────────────────────────────────────── */
        .tip-row {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            margin-bottom: 10px;
        }
        .tip-number {
            width: 24px; height: 24px;
            border-radius: 50%;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
            flex-shrink: 0;
        }

        /* ── History entry ─────────────────────────────────── */
        .history-entry {
            display: flex;
            align-items: center;
            gap: 12px;
            background: #ffffff;
            border-radius: 14px;
            padding: 14px 16px;
            border: 1px solid rgba(15,35,64,0.08);
            box-shadow: 0 1px 4px rgba(15,35,64,0.05);
            margin-bottom: 8px;
        }

        /* ── Result hero band ─────────────────────────────── */
        .result-hero {
            padding: 20px 20px 40px;
            margin: -1rem -1rem 0;
            position: relative;
            overflow: hidden;
        }
        .result-hero.feed-now   { background: #0e7a3e; }
        .result-hero.reduce     { background: #b45309; }
        .result-hero.stop       { background: #b91c1c; }

        .result-hero .back-btn {
            background: rgba(255,255,255,0.15);
            border: none;
            border-radius: 8px;
            padding: 8px 12px;
            cursor: pointer;
            color: #ffffff;
            font-size: 13px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 18px;
        }
        .result-hero .hero-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            opacity: 0.65;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .result-hero .hero-title {
            font-family: 'Roboto Slab', serif;
            font-size: 26px;
            font-weight: 700;
            color: #ffffff;
        }

        /* ── Bottom navigation ────────────────────────────── */
        .pondiq-nav {
            position: sticky;
            bottom: 0;
            z-index: 100;
            background: #ffffff;
            border-top: 1px solid rgba(15,35,64,0.1);
            box-shadow: 0 -4px 16px rgba(15,35,64,0.08);
            padding: 8px 16px 12px;
            margin: 0 -1rem;
        }

        /* ── Responsive breakpoints ────────────────────────── */
        @media (max-width: 480px) {
            .block-container { padding: 0.5rem 0.5rem 1.5rem !important; }
            .decision-banner .title { font-size: 20px; }
        }

        @media (min-width: 768px) {
            .block-container { max-width: 640px !important; }
        }

        @media (min-width: 1024px) {
            .block-container { max-width: 720px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── API Client ───────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"


def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_predict(data: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/predict", json=data, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Navigation ───────────────────────────────────────────────────────
def render_nav() -> str:
    """Return the active page key. Renders a bottom nav on mobile-like layout."""
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True,
                     type="primary" if st.session_state.page == "welcome" else "secondary"):
            st.session_state.page = "welcome"
            st.rerun()
    with col2:
        if st.button("🧪 Check Pond", key="nav_check", use_container_width=True,
                     type="primary" if st.session_state.page == "entry" else "secondary"):
            st.session_state.page = "entry"
            st.rerun()
    with col3:
        if st.button("📋 History", key="nav_history", use_container_width=True,
                     type="primary" if st.session_state.page == "history" else "secondary"):
            st.session_state.page = "history"
            st.rerun()
    return st.session_state.get("page", "welcome")


# ── Page: Welcome ────────────────────────────────────────────────────
def render_welcome() -> None:
    # Hero image
    hero_url = (
        "https://images.unsplash.com/photo-1649347173558-a305d7b8ff98"
        "?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800"
    )
    fish_url = (
        "https://images.unsplash.com/photo-1607629194532-53c98b8180da"
        "?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=800"
    )

    # Hero section
    st.image(hero_url, use_container_width=True)
    st.markdown(
        """
        <div style="text-align:center; margin:-0.5rem 0 1rem;">
            <span style="font-size:20px;">🐟</span>
            <span style="font-family:'Roboto Slab',serif;font-weight:700;font-size:16px;color:#1a5fa8;"> PondIQ</span>
        </div>
        <h2 style="text-align:center;">Smarter pond management starts here.</h2>
        <p style="text-align:center;color:#4a6b8a;">Instant feeding decisions from 6 water readings.</p>
        """,
        unsafe_allow_html=True,
    )

    # What is PondIQ card
    with st.container():
        st.markdown('<div class="pondiq-card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px;font-weight:700;color:#4a6b8a;text-transform:uppercase;letter-spacing:1.5px;">'
            'What is PondIQ?</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "PondIQ is a **professional feeding advisor** for fish farmers. "
            "Enter 6 water quality readings from your pond and instantly receive "
            "a science-based recommendation on whether to feed your fish."
        )
        st.markdown(
            "Overfeeding wastes resources and degrades water quality. "
            "PondIQ helps you make the right call — every single day."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # How it works with fish image
    st.image(fish_url, use_container_width=True)
    st.markdown(
        '<p style="font-size:11px;font-weight:700;color:#4a6b8a;text-transform:uppercase;letter-spacing:1px;">'
        'How it works</p>',
        unsafe_allow_html=True,
    )
    steps = [
        "Measure 6 water parameters",
        "Enter readings into PondIQ",
        "Receive your feeding decision",
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(
            f'<span style="display:inline-flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<span style="width:20px;height:20px;border-radius:50%;background:#1a5fa8;'
            f'color:#fff;font-size:11px;display:inline-flex;align-items:center;'
            f'justify-content:center;">{i}</span>'
            f'<span style="font-size:13px;color:#0f2340;">{step}</span></span><br>',
            unsafe_allow_html=True,
        )

    # Parameters measured
    st.markdown("---")
    st.markdown(
        '<p style="font-size:11px;font-weight:700;color:#4a6b8a;text-transform:uppercase;letter-spacing:1.5px;">'
        'Parameters measured</p>',
        unsafe_allow_html=True,
    )
    params = [
        ("💧", "Dissolved Oxygen (DO)"),
        ("⚗️", "pH Level"),
        ("🧪", "Ammonia (mg/L)"),
        ("🌡️", "Temperature"),
        ("🔬", "Nitrate (PPM)"),
        ("🌊", "Turbidity"),
    ]
    cols = st.columns(2)
    for i, (emoji, label) in enumerate(params):
        with cols[i % 2]:
            st.markdown(
                f'<div class="pondiq-card" style="padding:10px 12px;">'
                f'<span style="font-size:15px;">{emoji}</span> '
                f'<span style="font-size:12px;color:#0f2340;font-weight:500;">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # CTA
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Start Pond Check →", use_container_width=True, type="primary"):
        st.session_state.page = "entry"
        st.rerun()
    st.caption("Free to use · Works on basic mobile data · No account needed")


# ── Page: Data Entry & Prediction ────────────────────────────────────
READINGS = [
    {"id": "do",          "label": "Dissolved Oxygen (DO)", "emoji": "💧", "min": 0.0,
     "max": 14.0, "step": 0.1,  "unit": " mg/L", "good": "5–9 mg/L is ideal"},
    {"id": "ph",          "label": "pH Level",              "emoji": "⚗️", "min": 4.0,
        "max": 10.0, "step": 0.1,  "unit": "",       "good": "6.5–8.5 is optimal"},
    {"id": "ammonia",     "label": "Ammonia (mg/L)",        "emoji": "🧪", "min": 0.0,
     "max": 5.0,  "step": 0.05, "unit": " mg/L", "good": "Below 0.5 mg/L is safe"},
    {"id": "temperature", "label": "Temperature",           "emoji": "🌡️", "min": 10.0,
        "max": 40.0, "step": 0.5,  "unit": "°C",    "good": "26–32°C is best"},
    {"id": "nitrate",     "label": "Nitrate (PPM)",         "emoji": "🔬", "min": 0.0,
     "max": 200.0, "step": 1.0,  "unit": " ppm",  "good": "Below 40 ppm is safe"},
    {"id": "turbidity",   "label": "Turbidity",             "emoji": "🌊", "min": 1.0,
        "max": 5.0,  "step": 1.0,  "unit": "",       "good": "2–3 = slightly green (ideal)"},
]

TURBIDITY_LABELS = {1: "Crystal Clear", 2: "Slight Colour",
                    3: "Green-Tinted", 4: "Cloudy", 5: "Very Turbid"}


DEFAULTS: dict[str, float] = {
    "do": 6.5, "ph": 7.2, "ammonia": 0.2,
    "temperature": 28.0, "nitrate": 20.0, "turbidity": 2.0,
}


def _render_slider(reading: dict) -> None:
    """Render a single parameter slider card."""
    key = reading["id"]

    # Initialise session state
    if key not in st.session_state:
        st.session_state[key] = DEFAULTS[key]

    # Card wrapper
    with st.container():
        st.markdown(
            f'<div class="param-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:18px;">{reading["emoji"]}</span>'
            f'<span style="font-weight:600;color:#0f2340;font-size:14px;flex:1;margin-left:8px;">{reading["label"]}</span>'
            f'</div>'
            f'<p style="color:#4a6b8a;font-size:12px;margin:4px 0 8px 26px;">{reading["good"]}</p>',
            unsafe_allow_html=True,
        )

        # Value display
        current_val = st.session_state[key]
        decimals = 2 if reading["step"] < 0.1 else (
            1 if reading["step"] < 1 else 0)
        display_val = f"{current_val:.{decimals}f}{reading['unit']}"
        turb_label = TURBIDITY_LABELS.get(
            int(current_val), "") if key == "turbidity" else ""
        label_extra = (
            f' <span style="background:#daeaf8;border-radius:6px;padding:2px 8px;font-size:11px;">'
            f'{turb_label}</span>' if turb_label else ""
        )

        st.markdown(
            f'<span style="font-family:Roboto Slab,serif;font-size:24px;font-weight:700;color:#1a5fa8;">'
            f'{display_val}</span>{label_extra}',
            unsafe_allow_html=True,
        )

        # Slider — key matches session state variable name
        st.slider(
            label=reading["label"],
            min_value=reading["min"],
            max_value=reading["max"],
            step=reading["step"],
            key=key,
            label_visibility="collapsed",
        )

        col_min, col_max = st.columns(2)
        with col_min:
            st.caption(f"{reading['min']}{reading['unit']}")
        with col_max:
            st.markdown(
                f'<p style="text-align:right;color:#4a6b8a;font-size:11px;margin:0;">'
                f'{reading["max"]}{reading["unit"]}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


def render_entry() -> None:
    st.markdown("## 🧪 Water Quality Readings")
    st.caption("Enter all 6 parameters, then tap the button below.")

    # ── Handle reset BEFORE any widgets are created ──────────────────
    # Setting widget-bound session-state keys is only allowed *before*
    # the corresponding widget is instantiated in the current script run.
    if st.session_state.get("_reset_trigger"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state._reset_trigger = False
        st.session_state.pop("_last_result", None)

    # Check server health once
    health = api_health()
    server_ok = health is not None

    if server_ok:
        st.success("✅ PondIQ ML model connected")
    else:
        st.warning(
            "⚠️ ML model offline — make sure the API is running (`uv run pondiq_api.py`)")

    st.markdown("---")

    # Render sliders
    for reading in READINGS:
        _render_slider(reading)

    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col_act, col_reset = st.columns([3, 1])
    with col_act:
        predict_clicked = st.button(
            "✨ Get Recommendation & Prediction",
            use_container_width=True,
            type="primary",
        )
    with col_reset:
        reset_clicked = st.button(
            "🔄 Reset", key="entry_reset", use_container_width=True)

    if reset_clicked:
        st.session_state._reset_trigger = True
        st.rerun()

    # Prediction
    if predict_clicked:
        input_data = {
            "do": st.session_state.do,
            "ph": st.session_state.ph,
            "ammonia": st.session_state.ammonia,
            "temperature": st.session_state.temperature,
            "nitrate": st.session_state.nitrate,
            "turbidity": st.session_state.turbidity,
        }
        with st.spinner("Analysing with ML model…"):
            result = api_predict(input_data)

        if result:
            st.session_state._last_result = result
            st.session_state._last_input = input_data
            # ── Save to history ───────────────────────────
            now = datetime.now()
            hour_12 = now.hour % 12 or 12
            ampm = "AM" if now.hour < 12 else "PM"
            entry = {
                "date": now.strftime("%a, %d %b"),
                "time": f"{hour_12}:{now.strftime('%M')} {ampm}",
                "decision": "Feed Now" if result["label"] == "Prime Feed" else result["label"],
                "do": input_data["do"],
                "ph": input_data["ph"],
                "ammonia": input_data["ammonia"],
                "temp": input_data["temperature"],
                "nitrate": input_data["nitrate"],
                "turbidity": input_data["turbidity"],
            }
            if "_history" not in st.session_state:
                st.session_state._history = []
            st.session_state._history.insert(0, entry)
            # ───────────────────────────────────────────────
            st.session_state.page = "result"
            st.rerun()
        else:
            st.error(
                "Could not reach the ML model. Make sure the PondIQ API is running:\n\n"
                "```bash\ncd PondIQ-main && uv run pondiq_api.py\n```"
            )


def render_result_page() -> None:
    """Full-page prediction result screen."""
    result = st.session_state.get("_last_result")
    if not result:
        st.session_state.page = "entry"
        st.rerun()
        return

    label = result["label"]
    # Map API class name to display name
    if label == "Prime Feed":
        label = "Feed Now"
    confidence = result["confidence"]
    probs = result.get("probabilities", {})
    # Remap probability keys for display
    if "Prime Feed" in probs:
        probs["Feed Now"] = probs.pop("Prime Feed")
    warnings = result.get("warning_flags", [])

    configs = {
        "Feed Now":     {"css_class": "feed-now",  "icon": "✅", "tip_bg": "#0e7a3e"},
        "Reduce Feed":  {"css_class": "reduce",    "icon": "⚠️", "tip_bg": "#b45309"},
        "Halt Feeding": {"css_class": "stop",      "icon": "🚫", "tip_bg": "#b91c1c"},
    }
    cfg = configs.get(label, configs["Feed Now"])
    conf_pct = round(confidence * 100)

    tips_map = {
        "Feed Now": [
            "Feed at your standard daily ration",
            "Distribute feed evenly across the pond surface",
            "Observe fish feeding behaviour for the first 10 minutes",
            "Remove any uneaten feed after 30 minutes",
        ],
        "Reduce Feed": [
            "Feed at 50% of your normal daily ration",
            "Monitor for uneaten feed after 20 minutes and remove it",
            "Increase surface aeration if available",
            "Recheck DO and ammonia this evening",
        ],
        "Halt Feeding": [
            "Stop feeding immediately for at least 24–48 hours",
            "Increase aeration or add fresh water to dilute toxins",
            "Perform a 20–30% water exchange if possible",
            "Retest all parameters before resuming feeding",
        ],
    }

    next_check_map = {
        "Feed Now": "Tomorrow at the same time",
        "Reduce Feed": "This evening — retest DO and ammonia",
        "Halt Feeding": "Tomorrow morning — retest all 6 parameters",
    }

    reasons_map = {
        "Feed Now": (
            f"ML model confidence: {conf_pct}%. All water quality parameters are within "
            "acceptable ranges. Dissolved oxygen, pH, ammonia, and temperature support "
            "healthy fish metabolism. Feeding now will result in efficient feed conversion."
        ),
        "Reduce Feed": (
            f"ML model confidence: {conf_pct}%. Some parameters are outside the ideal range. "
            "A reduced feeding rate will maintain fish health without adding further load to the pond."
        ),
        "Halt Feeding": (
            f"ML model confidence: {conf_pct}%. Critical water quality parameters are outside "
            "safe ranges. Feeding under these conditions will worsen water quality and stress your fish."
        ),
    }

    # ── Hero band ───────────────────────────────────────────────
    st.markdown(
        f'<div class="result-hero {cfg["css_class"]}">'
        f'<div class="hero-label">Today\'s Recommendation</div>'
        f'<div class="hero-title">{cfg["icon"]} {label.upper()}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Back button
    if st.button("← Back to Readings", key="back_entry", use_container_width=False):
        st.session_state.page = "entry"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confidence ─────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:11px;font-weight:700;color:#4a6b8a;text-transform:uppercase;letter-spacing:1px;">'
        'Model Confidence</p>',
        unsafe_allow_html=True,
    )
    st.progress(confidence, text=f"{conf_pct}% confident — XGBoost model")

    if probs:
        st.markdown(
            '<p style="font-size:11px;font-weight:700;color:#4a6b8a;margin-top:10px;">'
            'Class Probabilities</p>',
            unsafe_allow_html=True,
        )
        for cls, prob in probs.items():
            pct = round(prob * 100, 1)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#4a6b8a;">'
                f'<span>{cls}</span><span style="font-weight:600;">{pct}%</span></div>',
                unsafe_allow_html=True,
            )

    if warnings:
        st.warning("\n\n".join(f"• {w}" for w in warnings))

    # ── Analysis card ──────────────────────────────────────────
    with st.container():
        st.markdown('<div class="pondiq-card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px;font-weight:700;color:#4a6b8a;text-transform:uppercase;letter-spacing:1px;">'
            'Analysis</p>',
            unsafe_allow_html=True,
        )
        st.markdown(reasons_map.get(label, reasons_map["Feed Now"]))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Action Steps ───────────────────────────────────────────
    tips = tips_map.get(label, tips_map["Feed Now"])
    with st.container():
        st.markdown('<div class="pondiq-card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px;font-weight:700;color:#4a6b8a;text-transform:uppercase;letter-spacing:1px;">'
            'Action Steps</p>',
            unsafe_allow_html=True,
        )
        for i, tip in enumerate(tips, 1):
            st.markdown(
                f'<div class="tip-row">'
                f'<div class="tip-number" style="background:{cfg["tip_bg"]};">{i}</div>'
                f'<span style="font-size:14px;color:#0f2340;line-height:1.55;">{tip}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Next check ─────────────────────────────────────────────
    nc = next_check_map.get(label, next_check_map["Feed Now"])
    st.info(f"**🕐 Next Check:** {nc}")

    # ── Actions ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Reading", key="result_new_reading", use_container_width=True):
            st.session_state._reset_trigger = True
            st.session_state.page = "entry"
            st.rerun()
    with col2:
        if st.button("🏠 Home", key="result_home", use_container_width=True):
            st.session_state.pop("_last_result", None)
            st.session_state.page = "welcome"
            st.rerun()


# ── Page: History ────────────────────────────────────────────────────
MOCK_HISTORY = [
    {"date": "Today",        "time": "7:12 AM", "decision": "Feed Now",   "do": 6.8,
        "ph": 7.1, "ammonia": 0.15, "temp": 28, "nitrate": 18, "turbidity": 2},
    {"date": "Yesterday",    "time": "6:58 AM", "decision": "Reduce Feed",  "do": 4.8,
        "ph": 7.4, "ammonia": 0.55, "temp": 31, "nitrate": 25, "turbidity": 3},
    {"date": "Mon, 26 May",  "time": "7:03 AM", "decision": "Feed Now",   "do": 7.2,
        "ph": 7.0, "ammonia": 0.10, "temp": 27, "nitrate": 14, "turbidity": 2},
    {"date": "Sun, 25 May",  "time": "7:30 AM", "decision": "Halt Feeding", "do": 2.8,
        "ph": 6.2, "ammonia": 2.20, "temp": 34, "nitrate": 60, "turbidity": 5},
    {"date": "Sat, 24 May",  "time": "6:50 AM", "decision": "Feed Now",   "do": 6.5,
        "ph": 7.3, "ammonia": 0.20, "temp": 27, "nitrate": 16, "turbidity": 2},
    {"date": "Fri, 23 May",  "time": "7:15 AM", "decision": "Feed Now",   "do": 7.0,
        "ph": 7.2, "ammonia": 0.12, "temp": 26, "nitrate": 12, "turbidity": 2},
    {"date": "Thu, 22 May",  "time": "7:00 AM", "decision": "Reduce Feed",  "do": 4.5,
        "ph": 7.6, "ammonia": 0.60, "temp": 30, "nitrate": 30, "turbidity": 4},
]


def _get_history() -> list[dict]:
    """Merge user predictions with mock fallback data."""
    saved = st.session_state.get("_history", [])
    return saved + MOCK_HISTORY


def render_history() -> None:
    st.markdown("## 📋 Reading History")

    history = _get_history()
    if not history:
        st.info("No readings yet. Run a pond check to see your history here.")
        return

    st.caption(f"{len(history)} reading{'s' if len(history) != 1 else ''}")

    # Stats
    feed_now = sum(1 for e in history if e["decision"] == "Feed Now")
    reduce = sum(1 for e in history if e["decision"] == "Reduce Feed")
    stop = sum(1 for e in history if e["decision"] == "Halt Feeding")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("✅ Feed Days", feed_now)
    with c2:
        st.metric("⚠️ Reduce Days", reduce)
    with c3:
        st.metric("🚫 Stop Days", stop)

    # Mini bar chart — DO levels (last 14 entries max)
    st.markdown("---")
    st.markdown(
        '<p style="font-size:11px;font-weight:700;color:#4a6b8a;text-transform:uppercase;letter-spacing:1px;">'
        '📊 DO Level History (mg/L)</p>',
        unsafe_allow_html=True,
    )
    chart_entries = list(reversed(history[-14:]))
    chart_data = pd.DataFrame(
        [{"Day": e["date"].split(",")[0][:3] if "," in e["date"] else e["date"][:3],
          "DO": e["do"], "Decision": e["decision"]}
         for e in chart_entries]
    )
    color_map = {"Feed Now": "#0e7a3e",
                 "Reduce Feed": "#b45309", "Halt Feeding": "#b91c1c"}
    st.bar_chart(chart_data, x="Day", y="DO", color="Decision")

    # History entries
    st.markdown("---")
    decision_cfg = {
        "Feed Now":   {"icon": "✅", "color": "#0e7a3e", "bg": "#e6f5ee"},
        "Reduce Feed":  {"icon": "⚠️", "color": "#b45309", "bg": "#fef3e2"},
        "Halt Feeding": {"icon": "🚫", "color": "#b91c1c", "bg": "#fef2f2"},
    }

    for entry in history:
        dc = decision_cfg[entry["decision"]]
        st.markdown(
            f'<div class="history-entry">'
            f'<div style="width:42px;height:42px;border-radius:10px;background:{dc["bg"]};'
            f'display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
            f'<span style="font-size:20px;">{dc["icon"]}</span></div>'
            f'<div style="flex:1;min-width:0;">'
            f'<p style="margin:0;font-weight:700;color:#0f2340;font-size:14px;">{entry["decision"]}</p>'
            f'<p style="margin:2px 0 0;color:#4a6b8a;font-size:12px;">{entry["date"]} · {entry["time"]}</p>'
            f'</div>'
            f'<div style="text-align:right;flex-shrink:0;">'
            f'<p style="margin:0;font-family:Roboto Slab,serif;font-weight:700;color:#1a5fa8;font-size:14px;">DO {entry["do"]}</p>'
            f'<p style="margin:0;color:#4a6b8a;font-size:11px;">pH {entry["ph"]} · {entry["temp"]}°C</p>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()

    # Init session state
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    page = st.session_state.page

    # Render active page
    if page == "welcome":
        render_welcome()
    elif page == "entry":
        render_entry()
    elif page == "result":
        render_result_page()
    elif page == "history":
        render_history()

    # Bottom navigation
    with st.container():
        st.markdown('<div class="pondiq-nav">', unsafe_allow_html=True)
        render_nav()
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
