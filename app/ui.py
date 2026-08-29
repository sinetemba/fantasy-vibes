"""Shared UI components, dark/night mode theming and reusable charts."""

import html
from typing import Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "primary": "#37003c",
    "secondary": "#00ff85",
    "accent": "#f8f8f8",
    "home_win": "#00ff85",
    "draw": "#f1c40f",
    "away_win": "#e74c3c",
    "gold": "#FFD700",
    "silver": "#C0C0C0",
    "bronze": "#CD7F32",
    "dark": "#0b0c10",
    "card": "#1f2833",
    "text": "#f5f6f7",
    "muted": "#a0a0a0",
}

DARK_CSS = """
<style>
    .main > div { padding: 0rem 1rem; }

    .epl-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #00ff85;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: 1px;
    }
    .epl-subtitle {
        font-size: 1rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
    }

    .card {
        background-color: #1f2833;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
        border: 1px solid #2c3e50;
        color: #f5f6f7;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #00ff85;
        margin-bottom: 1rem;
        border-bottom: 2px solid #37003c;
        padding-bottom: 0.5rem;
    }

    .metric-box {
        background: linear-gradient(135deg, #37003c, #00ff85);
        color: #0b0c10;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; }
    .metric-label { font-size: 0.8rem; font-weight: 600; }

    .prob-bar-container { margin: 0.5rem 0; }
    .prob-bar-label { font-size: 0.9rem; font-weight: 600; color: #f5f6f7; }
    .prob-bar {
        height: 24px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        color: #0b0c10;
        min-width: 36px;
    }

    .info-box, .warning-box, .error-box {
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #0b0c10;
    }
    .info-box { background-color: #d5f5e3; border-left: 4px solid #27ae60; }
    .warning-box { background-color: #fef9e7; border-left: 4px solid #f1c40f; }
    .error-box { background-color: #fdedec; border-left: 4px solid #e74c3c; }

    .disclaimer {
        font-size: 0.75rem;
        color: #7f8c8d;
        text-align: center;
        padding: 1rem;
        border-top: 1px solid #2c3e50;
        margin-top: 2rem;
    }

    /* Hide default sidebar page nav */
    [data-testid="stSidebarNav"] { display: none !important; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0c10 0%, #1f2833 100%) !important;
        border-right: 2px solid #37003c;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #f5f6f7 !important;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: rgba(255, 255, 255, 0.06) !important;
        color: #f5f6f7 !important;
        border: 1px solid rgba(0, 255, 133, 0.3) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(0, 255, 133, 0.2) !important;
        border-color: #00ff85 !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #00ff85 !important;
        color: #0b0c10 !important;
        border: 1px solid #00ff85 !important;
    }

    /* Group table (used for standings) */
    .group-table {
        width: 100%;
        border-collapse: collapse;
    }
    .group-table th {
        background-color: #37003c;
        color: #00ff85;
        padding: 0.5rem;
        font-size: 0.85rem;
        text-align: center;
    }
    .group-table td {
        padding: 0.4rem 0.5rem;
        border-bottom: 1px solid #2c3e50;
        text-align: center;
        font-size: 0.85rem;
        color: #f5f6f7;
    }
    .group-table tr:nth-child(even) { background-color: #161b22; }
    .qualified { color: #00ff85; font-weight: 700; }
    .relegation { color: #e74c3c; font-weight: 700; }
</style>
"""


def safe_html(text: str) -> str:
    return html.escape(str(text))


def apply_theme():
    """Apply the dark/night mode CSS."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def display_header(title: str, subtitle: str):
    st.markdown(f'<div class="epl-title">{safe_html(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="epl-subtitle">{safe_html(subtitle)}</div>', unsafe_allow_html=True)


def display_disclaimer():
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <strong>Disclaimer:</strong> Predictions and stats are for entertainment only. "
        "This app is not affiliated with any football league or governing body. "
        "Data is sourced from free public APIs and datasets."
        "</div>",
        unsafe_allow_html=True,
    )


def display_metric_card(label: str, value: str, delta: Optional[str] = None):
    delta_html = f'<div style="font-size:0.75rem; opacity:0.85;">{safe_html(delta)}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">{safe_html(value)}</div>
            <div class="metric-label">{safe_html(label)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_info_message(message: str):
    st.markdown(f'<div class="info-box">ℹ️ {safe_html(message)}</div>', unsafe_allow_html=True)


def show_warning_message(message: str):
    st.markdown(f'<div class="warning-box">⚠️ {safe_html(message)}</div>', unsafe_allow_html=True)


def show_error_message(message: str):
    st.markdown(f'<div class="error-box">❌ {safe_html(message)}</div>', unsafe_allow_html=True)


def probability_bar(label: str, probability: float, color: str, max_width: int = 100):
    pct = probability * 100
    width = max(5, min(pct, max_width))
    st.markdown(
        f"""
        <div class="prob-bar-container">
            <div class="prob-bar-label">{safe_html(label)}</div>
            <div class="prob-bar" style="width: {width}%; background-color: {color};">
                {pct:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_outcome_pie_chart(home_win: float, draw: float, away_win: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Home Win", "Draw", "Away Win"],
                values=[home_win * 100, draw * 100, away_win * 100],
                marker=dict(colors=[COLORS["home_win"], COLORS["draw"], COLORS["away_win"]]),
                textinfo="label+percent",
                textfont=dict(size=14, color="#0b0c10"),
                hole=0.4,
                hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )
    return fig


def create_outcome_bars(home_win: float, draw: float, away_win: float) -> go.Figure:
    labels = ["Away Win", "Draw", "Home Win"]
    values = [away_win * 100, draw * 100, home_win * 100]
    colors = [COLORS["away_win"], COLORS["draw"], COLORS["home_win"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                textfont=dict(size=13, color=COLORS["text"]),
                hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=180,
        xaxis=dict(range=[0, 100], title="Probability (%)", gridcolor="#2c3e50"),
        yaxis=dict(gridcolor="#2c3e50"),
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        bargap=0.3,
    )
    return fig


def create_scoreline_chart(prob_matrix: np.ndarray, max_goals: int = 5) -> go.Figure:
    matrix = prob_matrix[: max_goals + 1, : max_goals + 1]
    text = [[f"{matrix[i][j]*100:.1f}%" for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix * 100,
            x=[str(i) for i in range(max_goals + 1)],
            y=[str(i) for i in range(max_goals + 1)],
            colorscale="Greens",
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10, color="black"),
            hovertemplate="Home: %{x} goals<br>Away: %{y} goals<br>Prob: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Scoreline Probability Matrix (%)",
        xaxis_title="Home Goals",
        yaxis_title="Away Goals",
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )
    return fig


def create_standings_table(rows: List[Dict]) -> str:
    body = ""
    for row in rows:
        cls = ""
        if row.get("position", 99) <= 4:
            cls = "qualified"
        elif row.get("position", 0) >= 18:
            cls = "relegation"
        body += f"""
        <tr>
            <td>{row.get('position', '-')}</td>
            <td class="{cls}" style="text-align:left; padding-left:0.5rem;">{safe_html(row.get('team', ''))}</td>
            <td>{row.get('played', 0)}</td>
            <td>{row.get('won', 0)}</td>
            <td>{row.get('drawn', 0)}</td>
            <td>{row.get('lost', 0)}</td>
            <td>{row.get('goals_for', 0)}</td>
            <td>{row.get('goals_against', 0)}</td>
            <td>{row.get('goal_difference', 0):+d}</td>
            <td style="font-weight:700;">{row.get('points', 0)}</td>
        </tr>
        """
    return f"""
    <table class="group-table">
        <thead>
            <tr>
                <th>#</th>
                <th style="text-align:left;">Team</th>
                <th>P</th>
                <th>W</th>
                <th>D</th>
                <th>L</th>
                <th>GF</th>
                <th>GA</th>
                <th>GD</th>
                <th>Pts</th>
            </tr>
        </thead>
        <tbody>{body}</tbody>
    </table>
    """


def match_card_html(
    home_team: str,
    away_team: str,
    home_score: Optional[int],
    away_score: Optional[int],
    status: str,
    minutes: str,
    round_label: str = "",
    time: str = "",
) -> str:
    score = (
        f"{home_score} - {away_score}"
        if home_score is not None and away_score is not None
        else "vs"
    )
    status_color = {
        "live": "#e74c3c",
        "half_time": "#f1c40f",
        "full_time": "#00ff85",
        "scheduled": "#3498db",
        "postponed": "#95a5a6",
        "cancelled": "#7f8c8d",
    }.get(status, "#3498db")
    status_label = minutes or status.replace("_", " ").upper()

    return f"""
    <div class="card" style="padding:1rem; margin:0.5rem 0;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
            <span style="font-size:0.8rem; color:#a0a0a0;">{safe_html(round_label)}</span>
            <span style="background:{status_color}; color:#0b0c10; padding:0.25rem 0.6rem; border-radius:12px; font-size:0.75rem; font-weight:700;">
                {safe_html(status_label)}
            </span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
            <div style="flex:1; text-align:right; font-weight:700; font-size:1.1rem;">{safe_html(home_team)}</div>
            <div style="min-width:80px; text-align:center;">
                <div style="font-size:1.8rem; font-weight:800; color:#00ff85;">{score}</div>
                <div style="font-size:0.75rem; color:#a0a0a0;">{safe_html(time)} SAST</div>
            </div>
            <div style="flex:1; text-align:left; font-weight:700; font-size:1.1rem;">{safe_html(away_team)}</div>
        </div>
    </div>
    """
