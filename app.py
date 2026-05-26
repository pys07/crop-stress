"""
Streamlit Web Application - Crop Stress Prediction
===================================================
Interactive dashboard for model training, comparison, and field-level predictions.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from urllib.parse import quote

from src.config import MODEL_LABELS, SOIL_PARAMETER_COLUMNS, TARGET_COLUMNS, MODEL_CHARACTERISTICS, REPORTS_DIR
from src.ml_utils import (
    get_feature_ranges,
    load_dataset,
    load_report,
    predict_stress,
    train_and_save_models,
)
from src.evaluation import (
    create_summary_report,
    plot_model_comparison,
    plot_target_metrics,
    plot_confusion_matrices,
)

st.set_page_config(page_title="Crop Stress Prediction", layout="wide")

ASSET_PLANT_SVG = "assets/plant.svg"

THEMES = {
    "Light": {
        "background": """
            linear-gradient(180deg, rgba(249, 246, 240, 0.98), rgba(244, 246, 240, 0.96)),
            radial-gradient(circle at 12% 18%, rgba(76, 154, 111, 0.18), transparent 55%),
            radial-gradient(circle at 88% 12%, rgba(26, 74, 74, 0.12), transparent 50%),
            linear-gradient(90deg, rgba(26, 74, 74, 0.05) 1px, transparent 1px),
            linear-gradient(180deg, rgba(26, 74, 74, 0.05) 1px, transparent 1px)
        """,
        "text": "#1B2E2C",
        "muted": "#546561",
        "panel": "rgba(255, 255, 255, 0.78)",
        "panel_border": "rgba(43, 94, 59, 0.12)",
        "hero_a": "rgba(43, 94, 59, 0.96)",
        "hero_b": "rgba(76, 154, 111, 0.92)",
        "hero_text": "#f8f6ed",
        "badge_bg": "rgba(43, 94, 59, 0.12)",
        "badge_text": "#2B5E3B",
        "accent": "#2B5E3B",
        "chart_text": "#1A4A4A",
        "grid": "#E6E0D6",
    },
    "Dark": {
        "background": """
            radial-gradient(circle at top left, rgba(91, 163, 104, 0.18), transparent 24%),
            radial-gradient(circle at top right, rgba(220, 177, 49, 0.12), transparent 22%),
            linear-gradient(180deg, #0e1711 0%, #132219 56%, #18291e 100%)
        """,
        "text": "#ecf2ea",
        "muted": "#a8b7ac",
        "panel": "rgba(20, 34, 25, 0.76)",
        "panel_border": "rgba(157, 189, 166, 0.18)",
        "hero_a": "rgba(34, 76, 48, 0.95)",
        "hero_b": "rgba(126, 143, 64, 0.92)",
        "hero_text": "#f4f7ef",
        "badge_bg": "#203727",
        "badge_text": "#dbeed9",
        "accent": "#4C9A6F",
        "chart_text": "#e6efe5",
        "grid": "#35523d",
    },
}

RISK_COLORS = {
    "High": "#E67E22",
    "Moderate": "#D9A441",
    "Low": "#2f8f57",
}

KEY_PREDICTION_FEATURES = {
    "air_temperature_max",
    "air_temperature_min",
    "soil_moisture_10cm",
    "soil_moisture_30cm",
    "soil_pH",
    "day_of_year",
}

RISK_RANK = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
}


@st.cache_data
def get_dataset():
    return load_dataset()


@st.cache_data
def get_report():
    return load_report()


@st.cache_data
def load_svg_data_uri(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        svg_text = file.read()
    return f"data:image/svg+xml;utf8,{quote(svg_text)}"


def prettify(name: str) -> str:
    return name.replace("_", " ").replace("VPD", "VPD").title()


def get_primary_model_name(report: dict, preferred: str = "bilstm") -> str:
    models = report.get("models", {})
    if preferred in models:
        return preferred

    best_model_name = report.get("best_model", {}).get("name")
    if best_model_name in models:
        return best_model_name

    if models:
        return next(iter(models))

    raise ValueError("No trained models available in the report.")


def get_theme_name() -> str:
    if "theme_name" not in st.session_state:
        st.session_state.theme_name = "Light"
    return st.session_state.theme_name


def inject_styles(theme_name: str):
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        .stApp {{
            background: {theme["background"]};
            color: {theme["text"]};
            font-family: "Poppins", "Segoe UI", sans-serif;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320" viewBox="0 0 320 320" fill="none"><path d="M24 280C88 230 152 210 216 220C252 226 284 242 304 260" stroke="%232B5E3B" stroke-opacity="0.08" stroke-width="3" stroke-linecap="round"/><path d="M16 40C62 78 110 92 160 86C204 80 248 56 296 16" stroke="%231A4A4A" stroke-opacity="0.08" stroke-width="2" stroke-linecap="round"/></svg>');
            background-repeat: no-repeat;
            background-position: right -60px top 80px;
        }}
        h1, h2, h3 {{
            font-family: "Playfair Display", "Times New Roman", serif;
            letter-spacing: 0.02em;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .hero-panel {{
            padding: 2rem 2.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, {theme["hero_a"]}, {theme["hero_b"]});
            color: {theme["hero_text"]};
            box-shadow: 0 26px 56px rgba(19, 32, 28, 0.2);
            margin-bottom: 1.25rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-panel::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, rgba(255, 255, 255, 0.1), transparent 55%);
            pointer-events: none;
        }}
        .hero-plant {{
            position: absolute;
            right: 2rem;
            top: 50%;
            transform: translateY(-50%);
            width: 120px;
            height: 120px;
            object-fit: contain;
            opacity: 0.9;
            filter: drop-shadow(0 10px 20px rgba(0, 0, 0, 0.25));
        }}
        .hero-stress {{
            position: absolute;
            right: 2rem;
            top: 50%;
            transform: translateY(-50%);
            width: 140px;
            height: 140px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.25);
            background: conic-gradient(rgba(230, 126, 34, 0.9) 0 120deg, rgba(76, 154, 111, 0.9) 120deg 240deg, rgba(26, 74, 74, 0.8) 240deg 360deg);
            display: grid;
            place-items: center;
            animation: gaugePulse 3.5s ease-in-out infinite;
        }}
        .hero-stress::after {{
            content: "";
            width: 96px;
            height: 96px;
            border-radius: 50%;
            background: rgba(14, 23, 17, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(6px);
        }}
        .hero-stress img {{
            width: 72px;
            height: 72px;
            object-fit: contain;
            position: relative;
            z-index: 1;
            filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.25));
        }}
        @keyframes gaugePulse {{
            0% {{ transform: translateY(-50%) scale(1); }}
            50% {{ transform: translateY(-50%) scale(1.04); }}
            100% {{ transform: translateY(-50%) scale(1); }}
        }}
        .glass-card {{
            padding: 1.1rem 1.2rem;
            border-radius: 16px;
            background: {theme["panel"]};
            border: 1px solid {theme["panel_border"]};
            box-shadow: 0 12px 28px rgba(19, 32, 28, 0.12);
            backdrop-filter: blur(6px);
        }}
        .tooltip-card {{
            background: rgba(26, 74, 74, 0.9) !important;
            color: #f9f6f0 !important;
            border-radius: 10px !important;
            padding: 8px 12px !important;
            font-size: 0.85rem !important;
        }}
        .stat-card {{
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: {theme["panel"]};
            border: 1px solid {theme["panel_border"]};
            box-shadow: 0 12px 24px rgba(19, 32, 28, 0.12);
            min-height: 120px;
        }}
        .stat-label {{
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {theme["muted"]};
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {theme["text"]};
            margin-top: 0.2rem;
        }}
        .stat-note {{
            font-size: 0.95rem;
            color: {theme["muted"]};
            margin-top: 0.3rem;
        }}
        .section-tag {{
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: {theme["badge_bg"]};
            color: {theme["badge_text"]};
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }}
        .pill-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            align-items: center;
            margin: 0.6rem 0 1.2rem;
        }}
        .pill-label {{
            font-size: 0.8rem;
            color: {theme["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .pill-badge {{
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            background: rgba(43, 94, 59, 0.12);
            color: {theme["accent"]};
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .stress-gauge-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
        }}
        .stress-gauge-card {{
            padding: 1.1rem;
            border-radius: 16px;
            background: {theme["panel"]};
            border: 1px solid {theme["panel_border"]};
            box-shadow: 0 12px 24px rgba(19, 32, 28, 0.12);
            text-align: center;
        }}
        .stress-gauge {{
            --value: 0;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            margin: 0 auto 0.8rem;
            background: conic-gradient(#4C9A6F calc(var(--value) * 1%), {theme["grid"]} 0);
            display: grid;
            place-items: center;
        }}
        .stress-gauge::after {{
            content: "";
            width: 86px;
            height: 86px;
            border-radius: 50%;
            background: {theme["panel"]};
            border: 1px solid {theme["panel_border"]};
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.08);
        }}
        .stress-gauge-value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {theme["text"]};
        }}
        .stress-gauge-label {{
            font-size: 0.85rem;
            color: {theme["muted"]};
        }}
        .model-highlight {{
            padding: 1.2rem;
            border-radius: 16px;
            border: 1px solid rgba(43, 94, 59, 0.2);
            background: linear-gradient(135deg, rgba(43, 94, 59, 0.08), rgba(76, 154, 111, 0.12));
            text-align: center;
        }}
                .side-panel {{
                    padding: 1.2rem;
                    border-radius: 16px;
                    border: 1px solid {theme["panel_border"]};
                    background: {theme["panel"]};
                    box-shadow: 0 12px 24px rgba(19, 32, 28, 0.12);
                }}
                .badge-row {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                    margin-top: 0.6rem;
                }}
                .badge-pill {{
                    padding: 0.3rem 0.7rem;
                    border-radius: 999px;
                    background: rgba(26, 74, 74, 0.12);
                    color: {theme["accent"]};
                    font-weight: 600;
                    font-size: 0.78rem;
                    letter-spacing: 0.03em;
                    text-transform: uppercase;
                }}
        .model-highlight h3 {{
            margin-bottom: 0.4rem;
        }}
        .model-highlight .pill-badge {{
            margin-top: 0.6rem;
            display: inline-block;
        }}
        .timeline {{
            display: grid;
            gap: 1.1rem;
            position: relative;
            padding-left: 1.5rem;
        }}
        .timeline::before {{
            content: "";
            position: absolute;
            left: 0.5rem;
            top: 0.2rem;
            bottom: 0.2rem;
            width: 2px;
            background: rgba(43, 94, 59, 0.2);
        }}
        .timeline-item {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.9rem;
        }}
        .timeline-marker {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: rgba(76, 154, 111, 0.2);
            color: {theme["accent"]};
            font-weight: 700;
            display: grid;
            place-items: center;
            border: 1px solid rgba(43, 94, 59, 0.3);
        }}
        .timeline-title {{
            font-weight: 600;
            font-size: 1rem;
            color: {theme["text"]};
        }}
        .timeline-desc {{
            color: {theme["muted"]};
            margin-top: 0.2rem;
        }}
        .nav-help {{
            color: {theme["muted"]};
            font-size: 0.92rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: {theme["text"]};
        }}
        div[data-testid="stSidebar"] {{
            background: {theme["panel"]};
        }}
        div[data-testid="stMetricValue"] {{
            font-weight: 700;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, #2B5E3B, #4C9A6F);
            color: white;
            border-radius: 12px;
            border: none;
            padding: 0.7rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 12px 24px rgba(19, 32, 28, 0.18);
        }}
        .stButton > button:hover {{
            filter: brightness(1.05);
        }}
        div[data-baseweb="select"] > div {{
            border-radius: 999px;
            border-color: {theme["panel_border"]};
            background: {theme["panel"]};
        }}
        div[data-baseweb="select"] span {{
            color: {theme["text"]};
            font-weight: 600;
        }}
        @media (max-width: 780px) {{
            .block-container {{
                padding-top: 1rem;
                padding-bottom: 1rem;
            }}
            .hero-panel {{
                padding: 1.4rem 1.5rem;
            }}
            .hero-plant {{
                display: none;
            }}
            div[data-testid="stSidebar"] {{
                min-width: 220px;
                width: 220px;
            }}
            .nav-help {{
                display: none;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_card(title: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{title}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav() -> str:
    """Render sidebar navigation."""
    with st.sidebar:
        st.markdown("## Navigation")
        page_options = ["Home", "Dashboard", "Predictor", "Model Lab", "Analysis", "Project Info"]
        # If another widget requested a navigation change earlier in the app,
        # apply it now before instantiating the radio widget to avoid Streamlit
        # complaining about modifying a widget-backed session key.
        if "goto_page" in st.session_state:
            st.session_state.nav_page = st.session_state.pop("goto_page")
        if "nav_page" not in st.session_state:
            st.session_state.nav_page = "Home"
        page = st.radio(
            "Go to",
            page_options,
            index=page_options.index(st.session_state.nav_page),
            key="nav_page",
            label_visibility="collapsed",
        )
        selected_theme = st.toggle("Dark mode", value=get_theme_name() == "Dark", key="dark_mode_toggle")
        st.session_state.theme_name = "Dark" if selected_theme else "Light"
        st.markdown(
            '<div class="nav-help">Navigate between overview, analytics, predictions, and project details.</div>',
            unsafe_allow_html=True,
        )
    return page


def compute_overview(df: pd.DataFrame, report: dict) -> dict:
    total_records = len(df)
    total_fields = int(df["field_id"].nunique()) if "field_id" in df.columns else 0
    if "date" in df.columns:
        date_min = df["date"].min()
        date_max = df["date"].max()
    else:
        date_min = None
        date_max = None
    if date_min is None or date_max is None or pd.isna(date_min) or pd.isna(date_max):
        date_range = "Date range unavailable"
        year_count = None
    else:
        date_range = f"{date_min.date()} to {date_max.date()}"
        year_count = max((date_max - date_min).days / 365.25, 0)
    stress_rates = {}
    for target in TARGET_COLUMNS:
        if target in df.columns:
            rate = float(df[target].mean() * 100)
            stress_rates[target] = rate if np.isfinite(rate) else 0.0
        else:
            stress_rates[target] = 0.0
    return {
        "total_records": total_records,
        "total_fields": total_fields,
        "date_range": date_range,
        "year_count": year_count,
        "best_model": report["best_model"]["label"],
        "best_score": report["best_model"]["average_f1_score"],
        "best_accuracy": report["best_model"].get("average_accuracy", report["best_model"]["average_f1_score"]),
        "stress_rates": stress_rates,
    }


def get_report_metadata() -> dict:
    report_path = REPORTS_DIR / "training_report.json"
    if not report_path.exists():
        return {"last_trained": "Unavailable"}
    last_trained = datetime.fromtimestamp(report_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return {"last_trained": last_trained}


def get_chart_theme(theme_name: str) -> alt.ThemeConfig:
    theme = THEMES[theme_name]
    return {
        "config": {
            "view": {"stroke": None},
            "background": "transparent",
            "title": {"color": theme["chart_text"], "fontSize": 16},
            "axis": {
                "labelColor": theme["chart_text"],
                "titleColor": theme["chart_text"],
                "gridColor": theme["grid"],
                "domainColor": theme["grid"],
                "tickColor": theme["grid"],
            },
            "legend": {
                "labelColor": theme["chart_text"],
                "titleColor": theme["chart_text"],
            },
        }
    }


def themed_chart(chart: alt.Chart, theme_name: str):
    st.altair_chart(chart.configure(**get_chart_theme(theme_name)["config"]), use_container_width=True)


def slider_value(details: dict) -> tuple[float, float, float]:
    minimum = float(round(details["min"], 2))
    maximum = float(round(details["max"], 2))
    if minimum == maximum:
        maximum = minimum + 0.01
    default = min(max(float(round(details["mean"], 2)), minimum), maximum)
    return minimum, maximum, default


def default_feature_value(details: dict) -> float:
    baseline = details.get("median", details.get("mean", 0.0))
    return float(round(baseline, 2))


def get_quantile(df: pd.DataFrame, column: str, quantile: float) -> float:
    if column not in df.columns:
        return 0.0
    return float(df[column].quantile(quantile))


def compute_alert_thresholds(df: pd.DataFrame) -> dict:
    return {
        "temp_high": get_quantile(df, "air_temperature_max", 0.95),
        "temp_warn": get_quantile(df, "air_temperature_max", 0.85),
        "temp_min_high": get_quantile(df, "air_temperature_min", 0.95),
        "temp_min_warn": get_quantile(df, "air_temperature_min", 0.85),
        "moisture_low": get_quantile(df, "soil_moisture_10cm", 0.1),
        "moisture_warn": get_quantile(df, "soil_moisture_10cm", 0.2),
        "moisture_deep_high": get_quantile(df, "soil_moisture_30cm", 0.9),
        "moisture_deep_warn": get_quantile(df, "soil_moisture_30cm", 0.8),
        "rain_high": get_quantile(df, "precipitation", 0.9),
        "rain_low": get_quantile(df, "precipitation", 0.1),
    }


def apply_rule_based_alerts(predictions: dict, payload: dict, thresholds: dict) -> dict:
    def set_alert(target: str, level: str, reason: str) -> None:
        current = predictions[target]["risk_level"]
        if RISK_RANK[level] > RISK_RANK[current]:
            predictions[target]["risk_level"] = level
            predictions[target]["alert_reason"] = reason
            predictions[target]["prediction"] = int(level in {"Moderate", "High"})

    temp_max = payload.get("air_temperature_max")
    soil_moisture = payload.get("soil_moisture_10cm")
    deep_moisture = payload.get("soil_moisture_30cm")
    rainfall = payload.get("precipitation")

    if temp_max is not None and temp_max >= thresholds["temp_high"]:
        set_alert("temperature_stress", "High", "Air temperature max is very high")
    elif temp_max is not None and temp_max >= thresholds["temp_warn"]:
        set_alert("temperature_stress", "Moderate", "Air temperature max is elevated")

    if (
        soil_moisture is not None
        and rainfall is not None
        and temp_max is not None
        and soil_moisture <= thresholds["moisture_low"]
        and rainfall <= thresholds["rain_low"]
        and temp_max >= thresholds["temp_warn"]
    ):
        set_alert("water_stress", "High", "Low soil moisture with high temperature and low rainfall")
    elif (
        soil_moisture is not None
        and rainfall is not None
        and temp_max is not None
        and soil_moisture <= thresholds["moisture_warn"]
        and rainfall <= thresholds["rain_low"]
        and temp_max >= thresholds["temp_warn"]
    ):
        set_alert("water_stress", "Moderate", "Low soil moisture with elevated temperature and low rainfall")

    if (
        deep_moisture is not None
        and rainfall is not None
        and temp_max is not None
        and deep_moisture >= thresholds["moisture_deep_high"]
        and rainfall >= thresholds["rain_high"]
        and temp_max <= thresholds["temp_warn"]
    ):
        set_alert("waterlogging_stress", "High", "High soil moisture with heavy rainfall and cooler temperatures")
    elif (
        deep_moisture is not None
        and rainfall is not None
        and temp_max is not None
        and deep_moisture >= thresholds["moisture_deep_warn"]
        and rainfall >= thresholds["rain_high"]
        and temp_max <= thresholds["temp_warn"]
    ):
        set_alert("waterlogging_stress", "Moderate", "High soil moisture with heavy rainfall and moderate temperatures")

    return predictions


def day_of_year_to_date(day_of_year: float) -> datetime.date:
    base_date = datetime(2024, 1, 1)
    offset = int(round(day_of_year)) - 1
    return (base_date + pd.to_timedelta(offset, unit="D")).date()


def build_input_form(feature_ranges: dict, selected_features: list[str]) -> dict:
    st.markdown('<div class="section-tag">Field Input</div>', unsafe_allow_html=True)
    st.subheader("Enter Soil and Weather Parameters")
    st.caption("Enter the key inputs below. Remaining model features are filled automatically from historical medians.")

    payload = {
        feature: default_feature_value(feature_ranges[feature])
        for feature in selected_features
        if feature in feature_ranges
    }
    soil_features = [feature for feature in selected_features if feature in SOIL_PARAMETER_COLUMNS]
    other_features = [feature for feature in selected_features if feature not in soil_features]

    left, right = st.columns(2)

    with left:
        st.markdown("### Soil Parameters")
        for feature in soil_features:
            if feature not in KEY_PREDICTION_FEATURES:
                continue
            minimum, maximum, default = slider_value(feature_ranges[feature])
            payload[feature] = st.slider(prettify(feature), min_value=minimum, max_value=maximum, value=default)

    with right:
        st.markdown("### Weather and Time Parameters")
        for feature in other_features:
            if feature == "day_of_year":
                minimum, maximum, default = slider_value(feature_ranges[feature])
                selected_date = st.date_input(
                    "Day of Year",
                    value=day_of_year_to_date(default),
                    min_value=datetime(2020, 1, 1).date(),
                    max_value=datetime(2030, 12, 31).date(),
                )
                payload[feature] = selected_date.timetuple().tm_yday
                continue
            if feature not in KEY_PREDICTION_FEATURES:
                continue
            minimum, maximum, default = slider_value(feature_ranges[feature])
            payload[feature] = st.slider(prettify(feature), min_value=minimum, max_value=maximum, value=default)

        if "precipitation" in feature_ranges:
            st.markdown("#### Rainfall")
            minimum, maximum, default = slider_value(feature_ranges["precipitation"])
            payload["precipitation"] = st.slider("Rainfall", min_value=minimum, max_value=maximum, value=default)

    return payload


def render_prediction_cards(predictions: dict):
    columns = st.columns(len(predictions))
    for column, (target, details) in zip(columns, predictions.items()):
        border = RISK_COLORS[details["risk_level"]]
        with column:
            alert_reason = details.get("alert_reason")
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 6px solid {border};">
                    <div class="stat-label">{prettify(target)}</div>
                    <div class="stat-value">{details['probability'] * 100:.1f}%</div>
                    <div class="stat-note">{details['risk_level']} risk</div>
                    <div class="stat-note">{'Stress predicted' if details['prediction'] else 'No stress predicted'}</div>
                    {f'<div class="stat-note">{alert_reason}</div>' if alert_reason else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_recommendations(predictions: dict):
    st.subheader("Suggested Actions")
    advice = []

    if predictions["temperature_stress"]["risk_level"] in {"High", "Moderate"}:
        advice.append("Temperature stress risk is elevated. Consider mulching, shade nets, and revisiting irrigation timing.")
    if predictions["water_stress"]["risk_level"] in {"High", "Moderate"}:
        advice.append("Water stress risk is elevated. Check root-zone moisture and plan timely irrigation support.")
    if predictions["waterlogging_stress"]["risk_level"] in {"High", "Moderate"}:
        advice.append("Waterlogging risk is elevated. Improve drainage and reduce excess watering where possible.")
    if not advice:
        advice.append("Current conditions look stable with low stress risk across the selected stress types.")

    for line in advice:
        st.write(f"- {line}")


def render_home_page(df: pd.DataFrame, report: dict, overview: dict):
    plant_svg = load_svg_data_uri(ASSET_PLANT_SVG)
    primary_model_name = get_primary_model_name(report)
    primary_label = report["models"][primary_model_name]["label"]
    coverage_range = "01-01-2020 to 31-12-2025"
    coverage_years = "5 years"
    st.markdown(
        f"""
        <div class="hero-panel">
            <h1 style="margin: 0 0 0.5rem 0;">Crop Stress Prediction Platform</h1>
            <p style="font-size: 1.08rem; max-width: 760px;">
                Early detection of crop stress using machine learning. This platform combines traditional ML models 
                with advanced neural networks (BiLSTM) to predict temperature stress, water stress, and waterlogging risk.
            </p>
            <img class="hero-plant" src="{plant_svg}" alt="Plant" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_card("Dataset Size", f"{overview['total_records']:,}", "Historical observations")
    with col2:
        show_card("Fields Monitored", str(overview["total_fields"]), "Distinct farm plots")
    with col3:
        show_card("Best Model", overview["best_model"], f"Accuracy: {overview['best_accuracy']:.3f}")
    with col4:
        show_card("Coverage", coverage_range, coverage_years)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("### Platform Features")
        st.write(
            f"• **Primary Model** - {primary_label} (fast, interpretable, production-ready)\n"
            "• **Multi-Model Comparison** - Naive Bayes, Linear Regression, and BiLSTM\n"
            "• **Real-time Predictions** - Stress probabilities per field condition\n"
            "• **Comprehensive Analytics** - Trends, metrics, and model performance"
        )

        st.markdown("### Model Highlight")
        st.markdown(
            f"""
            <div class="model-highlight">
                <h3>{overview["best_model"]}</h3>
                <div class="stat-value" style="font-size: 1.6rem; margin-top: 0.2rem;">Accuracy {overview["best_accuracy"]:.3f}</div>
                <div class="stat-note">Balanced accuracy and latency for production use.</div>
                <span class="pill-badge">Recommended</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Current Stress Levels")
        gauge_cards = "".join(
            f"<div class=\"stress-gauge-card\">"
            f"<div class=\"stress-gauge\" style=\"--value: {rate:.1f};\"></div>"
            f"<div class=\"stress-gauge-value\">{rate:.1f}%</div>"
            f"<div class=\"stress-gauge-label\">{prettify(target)}</div>"
            f"</div>"
            for target, rate in overview["stress_rates"].items()
        )
        st.markdown(f"<div class=\"stress-gauge-grid\">{gauge_cards}</div>", unsafe_allow_html=True)

    st.markdown("### Quick Start")
    st.write("1. Choose a model in Predictor. 2. Adjust soil and weather inputs. 3. Run a live prediction.")
    if st.button("Start Prediction", use_container_width=True):
        # Set a transient flag so navigation is applied before the sidebar
        # widget with key `nav_page` is instantiated (avoids Streamlit error).
        st.session_state.goto_page = "Predictor"
        st.rerun()


def render_dashboard_page(df: pd.DataFrame, report: dict, theme_name: str):
    st.markdown('<div class="section-tag">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.title("Field Analytics & Model Performance")

    overview = compute_overview(df, report)
    report_meta = get_report_metadata()
    st.caption(
        f"Dataset: {overview['total_records']:,} records | Fields: {overview['total_fields']} | "
        f"Coverage: {overview['date_range']} | Last trained: {report_meta['last_trained']}"
    )

    if "field_id" in df.columns:
        field_options = ["All Fields"] + sorted(df["field_id"].dropna().unique().tolist())
        st.markdown(
            """
            <div class="pill-group">
                <span class="pill-label">Field</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_field = st.selectbox("Filter by field", field_options, label_visibility="collapsed")
        filtered_df = df if selected_field == "All Fields" else df[df["field_id"] == selected_field]
    else:
        filtered_df = df

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("### Model Performance by Stress Type")
        metric_rows = []
        for model_name, details in report["models"].items():
            for target, target_metrics in details["targets"].items():
                metric_rows.append(
                    {
                        "Model": details["label"],
                        "Stress Type": prettify(target),
                        "F1-Score": target_metrics["f1_score"],
                    }
                )
        stacked_frame = pd.DataFrame(metric_rows)
        stacked_chart = (
            alt.Chart(stacked_frame)
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X("F1-Score:Q", stack="normalize", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Stress Type:N", sort="-x"),
                color=alt.Color("Model:N"),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Stress Type:N"),
                    alt.Tooltip("F1-Score:Q", format=".3f"),
                ],
            )
            .properties(height=300)
        )
        themed_chart(stacked_chart, theme_name)
        st.caption("Normalized stacked view of F1 contributions by model")

    with col2:
        st.markdown("### Stress Distribution")
        stress_rows = []
        for target in TARGET_COLUMNS:
            if target not in filtered_df.columns:
                rate = 0.0
            else:
                rate = float(filtered_df[target].mean() * 100)
            if not np.isfinite(rate):
                rate = 0.0
            stress_rows.append({"Stress Type": prettify(target), "Rate": rate})
        stress_frame = pd.DataFrame(stress_rows).fillna(0.0)

        if stress_frame["Rate"].sum() == 0:
            st.info("No stress distribution available for the selected filter.")
        else:
            donut_chart = (
                alt.Chart(stress_frame)
                .mark_arc(innerRadius=70, outerRadius=120, cornerRadius=8)
                .encode(
                    theta=alt.Theta("Rate:Q"),
                    color=alt.Color("Stress Type:N"),
                    tooltip=[alt.Tooltip("Stress Type:N"), alt.Tooltip("Rate:Q", format=".2f")],
                )
                .properties(height=300)
            )
            label_chart = (
                alt.Chart(stress_frame[stress_frame["Rate"] > 0])
                .mark_text(radius=130, size=12, fontWeight="bold")
                .encode(
                    theta=alt.Theta("Rate:Q"),
                    text=alt.Text("Rate:Q", format=".1f"),
                    color=alt.value(THEMES[theme_name]["chart_text"]),
                )
            )
            themed_chart(donut_chart + label_chart, theme_name)

    # Metrics comparison table
    st.markdown("### Target Metrics Snapshot")
    metric_rows = []
    for model_name, details in report["models"].items():
        for target, target_metrics in details["targets"].items():
            metric_rows.append(
                {
                    "Model": details["label"],
                    "Stress Type": prettify(target),
                    "Balanced Accuracy": target_metrics.get(
                        "balanced_accuracy", target_metrics["accuracy"]
                    ),
                    "Precision": target_metrics["precision"],
                    "Recall": target_metrics["recall"],
                    "F1-Score": target_metrics["f1_score"],
                }
            )
    metric_frame = pd.DataFrame(metric_rows)
    st.dataframe(metric_frame, use_container_width=True, hide_index=True)

    # Soil parameters
    st.markdown("### Soil Health Summary")
    soil_cols = [col for col in SOIL_PARAMETER_COLUMNS if col in filtered_df.columns]
    if soil_cols:
        soil_summary = filtered_df[soil_cols].agg(["mean", "min", "max"]).T.round(2)
        st.dataframe(soil_summary, use_container_width=True)


def render_predictor_page(df: pd.DataFrame, report: dict):
    """Render prediction page with enhanced model selection."""
    st.markdown('<div class="section-tag">Prediction Center</div>', unsafe_allow_html=True)
    st.title("Live Crop Stress Predictor")
    st.write("Enter soil and weather conditions to estimate crop stress probabilities.")

    selected_features = report["selected_features"]
    feature_ranges = get_feature_ranges(df, selected_features + ["precipitation"])

    # Enhanced model selection with information
    col1, col2 = st.columns([1.2, 0.9])

    with col1:
        st.markdown("### Select Prediction Model")
        model_options = list(report["models"].keys())
        default_model_name = get_primary_model_name(report)
        default_index = model_options.index(default_model_name) if default_model_name in model_options else 0
        model_name = st.radio(
            "Choose model",
            options=model_options,
            format_func=lambda key: f"{report['models'][key]['label']}",
            horizontal=True,
            index=default_index,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("### Model Snapshot")
        model_info = report["models"][model_name]
        avg_bal_acc = model_info.get("average_balanced_accuracy", model_info["average_accuracy"])
        st.metric("F1-Score", f"{model_info['average_f1_score']:.4f}")
        st.metric("Balanced Accuracy", f"{avg_bal_acc:.4f}")
        st.caption("F1 balances precision and recall; balanced accuracy accounts for class imbalance.")
        if model_name == report["best_model"]["name"]:
            st.markdown("<span class=\"pill-badge\">Best Model</span>", unsafe_allow_html=True)

    side_col, _ = st.columns([0.9, 1.5])
    with side_col:
        st.markdown("### Performance Panel")
        st.markdown(
            f"""
            <div class="side-panel">
                <div class="stat-label">Active model</div>
                <div class="stat-value" style="font-size: 1.5rem;">{report["models"][model_name]["label"]}</div>
                <div class="stat-note">Balanced accuracy and latency for live inference.</div>
                <div class="badge-row">
                    <span class="badge-pill">F1 {model_info['average_f1_score']:.3f}</span>
                    <span class="badge-pill">Bal Acc {avg_bal_acc:.3f}</span>
                    <span class="badge-pill">Latency 50ms</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model comparison section
    with st.expander("Compare All Models", expanded=False):
        comparison_df = pd.DataFrame([
            {
                "Model": report["models"][m]["label"],
                "F1-Score": report["models"][m]["average_f1_score"],
                "Balanced Accuracy": report["models"][m].get(
                    "average_balanced_accuracy", report["models"][m]["average_accuracy"]
                ),
                "Type": report["models"][m].get("type", "traditional").upper(),
            }
            for m in report["models"].keys()
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Model characteristics
        st.markdown("#### Model Characteristics")
        char_data = []
        for m in sorted(report["models"].keys()):
            if m in MODEL_CHARACTERISTICS:
                char = MODEL_CHARACTERISTICS[m]
                char_data.append({
                    "Model": report["models"][m]["label"],
                    "Speed": char["speed"],
                    "Accuracy": char["accuracy"],
                    "Interpretability": char["interpretability"],
                })
        if char_data:
            st.dataframe(pd.DataFrame(char_data), use_container_width=True, hide_index=True)

    st.markdown("### Active Soil Inputs")
    st.caption(", ".join(prettify(f) for f in report["soil_parameters"]))

    payload = build_input_form(feature_ranges, selected_features)
    alert_thresholds = compute_alert_thresholds(df)
    if st.button("Predict Stress", use_container_width=True):
        with st.spinner("Analyzing field conditions..."):
            predictions = predict_stress(model_name, payload)
            predictions = apply_rule_based_alerts(predictions, payload, alert_thresholds)
        st.markdown("### Prediction Results")
        render_prediction_cards(predictions)
        render_recommendations(predictions)


def render_model_lab_page(report: dict):
    """Render model lab page with enhanced analysis."""
    st.markdown('<div class="section-tag">Model Laboratory</div>', unsafe_allow_html=True)
    st.title("Model Comparison & Management")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        best_f1 = report["best_model"]["average_f1_score"]
        st.metric("Best F1-Score", f"{best_f1:.4f}")
    with col2:
        best_bal_acc = report["best_model"].get(
            "average_balanced_accuracy", report["best_model"]["average_accuracy"]
        )
        st.metric("Best Balanced Accuracy", f"{best_bal_acc:.4f}")
    with col3:
        model_count = len(report["models"])
        st.metric("Models Trained", str(model_count))
    with col4:
        feature_count = len(report["selected_features"])
        st.metric("Features Used", str(feature_count))

    # Detailed comparison
    st.markdown("### Comprehensive Model Comparison")
    col1, col2 = st.columns([1.5, 1])

    primary_model_name = get_primary_model_name(report)
    primary_details = report["models"][primary_model_name]
    
    with col1:
        comparison_df = pd.DataFrame([
            {
                "Model": details["label"],
                "Type": details.get("type", "traditional").upper(),
                "Avg F1": details["average_f1_score"],
                "Balanced Accuracy": details.get("average_balanced_accuracy", details.get("average_accuracy", 0)),
                "Status": "" if name == primary_model_name else ("BEST" if name == report["best_model"]["name"] else ""),
            }
            for name, details in report["models"].items()
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Primary Model")
        st.metric(
            "Recommended",
            primary_details["label"],
            f"F1: {primary_details['average_f1_score']:.2f}",
        )
        st.caption("Best for production use")
        st.caption("50ms predictions")
        st.caption("Great interpretability")

    # Model characteristics and use cases
    st.markdown("### Model Recommendations")
    char_data = []
    model_order = [primary_model_name] + [name for name in report["models"].keys() if name != primary_model_name]
    for model_name in model_order:
        if model_name in report["models"] and model_name in MODEL_CHARACTERISTICS:
            details = report["models"][model_name]
            char = MODEL_CHARACTERISTICS[model_name]
            char_data.append({
                "Model": (f"{details['label']}" if model_name == primary_model_name else details["label"]),
                "Best For": char["use_case"],
                "Speed": char["speed"],
                "Interpretability": char["interpretability"],
                "F1-Score": f"{details['average_f1_score']:.4f}",
            })
    
    if char_data:
        char_df = pd.DataFrame(char_data)
        st.dataframe(char_df, use_container_width=True, hide_index=True)

    # Retraining button
    st.markdown("### Model Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Retrain All Models", use_container_width=True, key="retrain_btn"):
            with st.spinner("Training models on full dataset..."):
                train_and_save_models()
                st.cache_data.clear()
            st.success("Models retrained successfully!")
            st.rerun()
    
    with col2:
        st.info("Tip: Retrain models after adding new data for better accuracy")

    # Selected features
    st.markdown("### Selected Features")
    st.caption(f"Total: {len(report['selected_features'])} features")
    features_col1, features_col2 = st.columns(2)
    
    soil_feats = [f for f in report["selected_features"] if f in SOIL_PARAMETER_COLUMNS]
    weather_feats = [f for f in report["selected_features"] if f not in soil_feats]
    
    with features_col1:
        st.write("**Soil Parameters:**")
        st.write(", ".join(prettify(f) for f in soil_feats) if soil_feats else "None")
    
    with features_col2:
        st.write("**Weather & Time:**")
        st.write(", ".join(prettify(f) for f in weather_feats) if weather_feats else "None")


def render_analysis_page(report: dict, df: pd.DataFrame):
    """Render detailed analysis page."""
    st.markdown('<div class="section-tag">Detailed Analysis</div>', unsafe_allow_html=True)
    st.title("Comprehensive Model Analysis")

    overview = compute_overview(df, report)
    report_meta = get_report_metadata()
    st.caption(
        f"Dataset: {overview['total_records']:,} records | Fields: {overview['total_fields']} | "
        f"Coverage: {overview['date_range']} | Last trained: {report_meta['last_trained']}"
    )

    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Model Comparison", "Target Metrics", "Feature Analysis", "Training Summary"],
    )

    if analysis_type == "Model Comparison":
        try:
            fig = plot_model_comparison(report, "average_f1_score")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render visualization: {str(e)}")
        
    elif analysis_type == "Target Metrics":
        model_name = st.selectbox(
            "Select Model",
            options=list(report["models"].keys()),
            format_func=lambda key: report["models"][key]["label"],
        )
        try:
            fig = plot_target_metrics(report, model_name)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render visualization: {str(e)}")

    elif analysis_type == "Feature Analysis":
        st.markdown("### Selected Features for Prediction")
        features_df = pd.DataFrame({
            "Feature": [prettify(f) for f in report["selected_features"]],
            "Type": ["Soil" if f in SOIL_PARAMETER_COLUMNS else "Weather" 
                    for f in report["selected_features"]]
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)

    else:  # Training Summary
        summary = create_summary_report(report)
        st.code(summary, language="text")


def render_scope_page(report: dict):
    """Render project info page."""
    st.markdown('<div class="section-tag">Project Information</div>', unsafe_allow_html=True)
    st.title("Project Overview & Workflow")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Project Objectives")
        st.write(
            "• Predict crop stress early using machine learning\n"
            "• Compare multiple model architectures\n"
            "• Support proactive farm management decisions\n"
            "• Integrate traditional ML with neural networks"
        )
    with col2:
        st.markdown("### Models Implemented")
        for model_name, details in report["models"].items():
            model_type = details.get("type", "traditional").upper()
            st.write(f"**{details['label']}** ({model_type})")

    st.markdown("### 🔄 ML Training Pipeline")
    st.markdown(
        """
        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-marker">1</div>
                <div>
                    <div class="timeline-title">Data Loading</div>
                    <div class="timeline-desc">Load crop stress dataset with weather and soil features</div>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">2</div>
                <div>
                    <div class="timeline-title">Preprocessing</div>
                    <div class="timeline-desc">Engineer calendar features (month, day of year, week)</div>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">3</div>
                <div>
                    <div class="timeline-title">Feature Selection</div>
                    <div class="timeline-desc">Rank features by importance, select top K features</div>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">4</div>
                <div>
                    <div class="timeline-title">Model Training</div>
                    <div class="timeline-desc">Train all models independently for each stress type</div>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">5</div>
                <div>
                    <div class="timeline-title">Evaluation</div>
                    <div class="timeline-desc">Calculate comprehensive metrics (F1, Balanced Accuracy, AUC)</div>
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-marker">6</div>
                <div>
                    <div class="timeline-title">Prediction</div>
                    <div class="timeline-desc">Use best model for live field predictions</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Training Configuration")
    config_df = pd.DataFrame({
        "Parameter": ["Test Size", "Features Used", "Soil Parameters", 
                     "BiLSTM Sequence Length", "Target Variables"],
        "Value": [
            f"{report.get('training_config', {}).get('test_size', 0.2) * 100:.0f}%",
            str(report.get('feature_count', len(report['selected_features']))),
            str(len(report['soil_parameters'])),
            "30 days",
            str(len(TARGET_COLUMNS))
        ]
    })
    st.dataframe(config_df, use_container_width=True, hide_index=True)


def main():
    page = render_top_nav()
    theme_name = get_theme_name()
    inject_styles(theme_name)

    df = get_dataset()
    report = get_report()
    overview = compute_overview(df, report)

    if page == "Home":
        render_home_page(df, report, overview)
    elif page == "Dashboard":
        render_dashboard_page(df, report, theme_name)
    elif page == "Predictor":
        render_predictor_page(df, report)
    elif page == "Model Lab":
        render_model_lab_page(report)
    elif page == "Analysis":
        render_analysis_page(report, df)
    else:
        render_scope_page(report)


if __name__ == "__main__":
    main()
