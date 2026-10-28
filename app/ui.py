"""Shared UI components, dark/night mode theming and reusable charts."""

import html
import logging
import threading
import time
from datetime import datetime
from textwrap import dedent
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data import LeagueService

logger = logging.getLogger(__name__)

# No TTL: an expiring entry makes the next visitor wait through a full
# fetch + refit. The background refresher below keeps services current.
@st.cache_resource(show_spinner=False)
def _cached_league_service(code: str) -> LeagueService:
    return LeagueService(code)


def get_league_service(code: str) -> LeagueService:
    """Shared LeagueService per league code, cached across reruns and sessions."""
    svc = _cached_league_service(code)
    # When Streamlit hot-reloads changed code, the cache still holds
    # instances of the old class, which lack any newly added methods.
    if not is_current_service(svc):
        _cached_league_service.clear()
        svc = _cached_league_service(code)
    return svc


def is_current_service(svc) -> bool:
    """False for services built from a class definition replaced by a reload."""
    return type(svc) is LeagueService


def prewarm_league_services(codes: List[str], limit: int = 4) -> None:
    """Build/cache LeagueService objects on daemon threads so first use
    (data fetch + model training) doesn't block the UI."""

    def _warm(code: str) -> None:
        try:
            get_league_service(code)
        except Exception:
            pass

    for code in list(codes)[:limit]:
        threading.Thread(target=_warm, args=(code,), daemon=True).start()


# Background refresh cadence — matches the current-season HTTP TTL, so each
# pass reloads standings and results that have actually expired.
BACKGROUND_REFRESH_SECONDS = 1800


def _refresh_loop() -> None:
    """Warm every configured league (dropdown + international group), then
    fully reload and refit each one every BACKGROUND_REFRESH_SECONDS. One
    sequential thread so it never blocks the UI or hammers providers;
    services swap new state in atomically, so readers keep the old data
    until a reload finishes."""
    from src.data.league_service import _leagues_registry
    from src.data.utils import bust_live_cache

    first = True
    while True:
        try:
            if first:
                # Bypass TTLs while warming so models train on the latest results.
                bust_live_cache(300)
            leagues = _leagues_registry()
            codes = [c for c, cfg in leagues.items() if not cfg.get("group")]
            codes += [c for c, cfg in leagues.items() if cfg.get("group")]
            for code in codes:
                try:
                    svc = get_league_service(code)
                    if first:
                        # A cached service may hold stale in-memory matches —
                        # pull fresh ones before refitting so the model trains
                        # on the latest results, not just a stale model file.
                        svc.refresh_live_matches()
                        svc.refit_model()
                    else:
                        svc.refresh(refit=True)
                except Exception as exc:
                    logger.warning(f"Background refresh failed for {code}: {exc}")
        except Exception as exc:
            logger.warning(f"Background refresh pass failed: {exc}")
        first = False
        time.sleep(BACKGROUND_REFRESH_SECONDS)


@st.cache_resource(show_spinner=False)
def _background_refresher() -> threading.Thread:
    # cache_resource makes this once per process (and survives Streamlit's
    # module reloads), so every session shares one refresh thread.
    thread = threading.Thread(target=_refresh_loop, daemon=True, name="league-refresh")
    thread.start()
    return thread


def prewarm_all_services() -> None:
    """Start the process-wide warm-up/refresh thread. Idempotent — safe to
    call from every session."""
    _background_refresher()


# A model whose newest training result is older than this is likely stale
# (e.g. a feed that stopped updating for a competition).
STALE_DATA_DAYS = 21
# Below this many training matches a prediction is mostly prior-driven.
LOW_DATA_MATCHES = 60


def staleness_note(pred: Dict) -> str:
    """Short 'low confidence' note when a prediction's underlying model data
    is stale or thin. Empty string when the data is fresh or unknown."""
    bits = []
    raw = pred.get("data_as_of")
    if raw:
        try:
            as_of = datetime.fromisoformat(raw)
            now = datetime.now(as_of.tzinfo) if as_of.tzinfo else datetime.now()
            age = (now - as_of).days
            if age > STALE_DATA_DAYS:
                bits.append(f"results only through {as_of.date()} ({age}d old)")
        except Exception:
            pass
    used = pred.get("matches_used") or 0
    if used and used < LOW_DATA_MATCHES:
        bits.append(f"only {used} matches")
    return " · ".join(bits)


def form_badges_html(form: List[Dict]) -> str:
    """Inline W/D/L badges (rightmost = latest) or an N/A placeholder."""
    if not form:
        return "<span style='color:#888;font-size:0.8rem;'>N/A</span>"
    colors = {"W": "#00ff85", "D": "#a0a0a0", "L": "#ff4b4b"}
    badges = []
    for m in form:
        result = m.get("result", "?")
        title = html.escape(
            f"{m.get('venue', '')} vs {m.get('opponent', '')} {m.get('score', '')}"
        )
        badges.append(
            f"<span title='{title}' "
            f"style='display:inline-block;width:1.4rem;height:1.4rem;line-height:1.4rem;"
            f"text-align:center;border-radius:4px;background:{colors.get(result, '#a0a0a0')}22;"
            f"color:{colors.get(result, '#a0a0a0')};font-weight:700;margin-right:4px;"
            f"font-size:0.8rem;'>{result}</span>"
        )
    return "".join(badges)


# Outcome colours mean one thing everywhere: which result (home / draw /
# away), never favourite vs underdog. Blue vs orange stays distinct under
# the common colour-vision deficiencies, where green vs red does not, and
# every segment is also labelled so colour is never the only cue.
OUTCOME_COLORS = {"home": "#3b8fd9", "draw": "#8a8f98", "away": "#f39c34"}

# Verdict thresholds on the gap between the two likeliest outcomes.
VERDICT_COIN_FLIP_GAP = 0.05
VERDICT_SLIGHT_GAP = 0.15
VERDICT_STRONG_GAP = 0.30


def _whole_percents(*probs: float) -> List[int]:
    """Probabilities as whole percents that still sum to 100 (largest
    remainder), so a bar never reads 48% + 26% + 27%."""
    raw = [max(0.0, float(p or 0)) * 100 for p in probs]
    total = sum(raw) or 1.0
    raw = [r * 100 / total for r in raw]
    floors = [int(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[: 100 - sum(floors)]:
        floors[i] += 1
    return floors


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _team_labels(pred: Dict, home: Optional[str], away: Optional[str]):
    return home or pred.get("home_team", "Home"), away or pred.get("away_team", "Away")


def prediction_verdict(pred: Dict, home: Optional[str] = None, away: Optional[str] = None) -> str:
    """Plain-language headline, e.g. 'Arsenal slight favourites' or
    'Coin flip', from the gap between the two likeliest outcomes."""
    home, away = _team_labels(pred, home, away)
    outcome = pred.get("outcome_probabilities") or {}
    ranked = sorted(
        (
            (outcome.get("home_win", 0), home),
            (outcome.get("draw", 0), None),
            (outcome.get("away_win", 0), away),
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    (top, team), (second, _) = ranked[0], ranked[1]
    gap = top - second
    if gap < VERDICT_COIN_FLIP_GAP:
        return "Coin flip"
    if team is None:
        return "Draw the likeliest result"
    if gap < VERDICT_SLIGHT_GAP:
        return f"{team} slight favourites"
    if gap < VERDICT_STRONG_GAP:
        return f"{team} favourites"
    return f"{team} strong favourites"


def top_scorelines(pred: Dict, n: int = 3) -> List[tuple]:
    """The n likeliest exact scores as ("1-0", probability) pairs."""
    matrix = pred.get("score_matrix")
    if matrix is None:
        score = pred.get("predicted_score") or {}
        return [(score["score"], score.get("probability", 0))] if score.get("score") else []
    m = np.asarray(matrix)
    cells = sorted(
        ((float(m[i][j]), f"{i}-{j}") for i in range(m.shape[0]) for j in range(m.shape[1])),
        reverse=True,
    )
    return [(score, p) for p, score in cells[:n]]


def scorelines_text(pred: Dict, n: int = 3) -> str:
    """'Most likely scores: 1-1 (12%) · 1-0 (11%) · 2-1 (9%)'."""
    scores = top_scorelines(pred, n)
    if not scores:
        return ""
    label = "Most likely score" if len(scores) == 1 else "Most likely scores"
    return f"{label}: " + " · ".join(f"{s} ({p * 100:.0f}%)" for s, p in scores)


def outcome_bar_html(pred: Dict, home: Optional[str] = None, away: Optional[str] = None) -> str:
    """One stacked home/draw/away bar with a labelled legend line, e.g.
    'Arsenal 48% · Draw 26% · Chelsea 26%'."""
    home, away = _team_labels(pred, home, away)
    outcome = pred.get("outcome_probabilities") or {}
    pcts = _whole_percents(
        outcome.get("home_win", 0), outcome.get("draw", 0), outcome.get("away_win", 0)
    )
    parts = list(zip((home, "Draw", away), pcts, ("home", "draw", "away")))
    legend = " · ".join(
        f"<span style='color:{OUTCOME_COLORS[key]};'>■</span> "
        f"{safe_html(label)} <b>{pct}%</b>"
        for label, pct, key in parts
    )
    segments = "".join(
        f"<div title='{safe_html(label)} {pct}%' style='width:{pct}%;"
        f"background:{OUTCOME_COLORS[key]};'></div>"
        for label, pct, key in parts
        if pct
    )
    return (
        f"<div style='font-size:0.9rem;margin:0.35rem 0 0.25rem;'>{legend}</div>"
        f"<div style='display:flex;height:12px;border-radius:6px;overflow:hidden;"
        f"gap:2px;background:#2c3e50;'>{segments}</div>"
    )


def _league_name(code: Optional[str]) -> Optional[str]:
    if not code:
        return code
    from src.data.league_service import _leagues_registry

    return _leagues_registry().get(code, {}).get("name", code)


def prediction_reasons(pred: Dict, home: Optional[str] = None, away: Optional[str] = None) -> str:
    """'Why: Arsenal 2nd in Premier League, 2.3 pts/game (last 10); …' —
    the table-position and form inputs behind the prediction."""
    home, away = _team_labels(pred, home, away)
    inputs = pred.get("model_inputs") or {}

    def _fmt(team: str, info: Dict) -> str:
        parts = []
        pos, league = info.get("position"), info.get("context_league")
        if pos and league:
            parts.append(f"{_ordinal(int(pos))} in {_league_name(league)}")
        if info.get("form_ppg") is not None:
            parts.append(f"{info['form_ppg']:.1f} pts/game (last 10)")
        comp_ppg, comp = info.get("comp_form_ppg"), info.get("competition")
        if comp_ppg is not None and comp and comp != league:
            parts.append(f"{comp_ppg:.1f} pts/game in {_league_name(comp)}")
        return f"{team} " + ", ".join(parts) if parts else ""

    bits = [b for b in (_fmt(home, inputs.get("home") or {}), _fmt(away, inputs.get("away") or {})) if b]
    return "Why: " + "; ".join(bits) if bits else ""


def prediction_caveat(pred: Dict) -> str:
    """Warning line when the prediction rests on thin or stale data —
    empty when it doesn't."""
    note = staleness_note(pred)
    if pred.get("model") == "elo_fallback":
        return "⚠️ Limited data: rating-only estimate" + (f" · {note}" if note else "")
    return f"⚠️ Low confidence — {note}" if note else ""


def _vs_average(ratio: float, verb: str) -> str:
    delta = round((float(ratio) - 1.0) * 100)
    if abs(delta) < 3:
        return f"{verb} about the league average"
    return f"{verb} {abs(delta)}% {'more' if delta > 0 else 'fewer'} than average"


def team_strength_lines(pred: Dict, home: Optional[str] = None, away: Optional[str] = None) -> List[str]:
    """Attack/defence ratings in words, venue-specific: 'Arsenal at home:
    scores 18% more than average, concedes 12% fewer than average'. Empty
    for rating-only fallback predictions, which have no such ratings."""
    if pred.get("model") == "elo_fallback":
        return []
    home, away = _team_labels(pred, home, away)
    attack = pred.get("team_attack_params") or {}
    defense = pred.get("team_defense_params") or {}
    return [
        f"**{team} {venue}:** {_vs_average(attack.get(side, 1.0), 'scores')}, "
        f"{_vs_average(defense.get(side, 1.0), 'concedes')}"
        for team, side, venue in ((home, "home", "at home"), (away, "away", "away"))
    ]


def prediction_summary_html(
    pred: Dict, home: Optional[str] = None, away: Optional[str] = None
) -> str:
    """Compact prediction block for fixture lists and match cards: verdict,
    outcome bar, likeliest scores, reasons and any data caveat."""
    reasons = prediction_reasons(pred, home, away)
    caveat = prediction_caveat(pred)
    muted = "font-size:0.8rem;color:#a0a0a0;margin-top:0.3rem;"
    return (
        f"<div style='font-weight:700;font-size:1rem;margin-top:0.5rem;'>"
        f"{safe_html(prediction_verdict(pred, home, away))}</div>"
        f"{outcome_bar_html(pred, home, away)}"
        f"<div style='font-size:0.85rem;margin-top:0.4rem;'>{safe_html(scorelines_text(pred))}</div>"
        + (f"<div style='{muted}'>{safe_html(reasons)}</div>" if reasons else "")
        + (f"<div style='{muted}color:#f1c40f;'>{safe_html(caveat)}</div>" if caveat else "")
    )


COLORS = {
    "primary": "#37003c",
    "secondary": "#00ff85",
    "accent": "#f8f8f8",
    "home_win": OUTCOME_COLORS["home"],
    "draw": OUTCOME_COLORS["draw"],
    "away_win": OUTCOME_COLORS["away"],
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

    /* Selected selectbox value and options */
    [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #000000 !important;
        font-weight: 900 !important;
        text-shadow: 0 0 2px #f5f6f7 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"][aria-selected="true"],
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"][aria-selected="true"] *,
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] * {
        color: #000000 !important;
        background-color: #00ff85 !important;
        font-weight: 900 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] [role="option"][aria-selected="true"],
    [data-testid="stSidebar"] [data-baseweb="select"] [role="option"][aria-selected="true"] * {
        color: #000000 !important;
        background-color: #00ff85 !important;
        font-weight: 900 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"],
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"] *,
    [data-testid="stSelectboxVirtualDropdown"] [role="option"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
    [data-testid="stSidebar"] [data-baseweb="select"] [role="option"],
    [data-testid="stSidebar"] [data-baseweb="select"] [role="option"] * {
        color: #f5f6f7 !important;
        background-color: #1f2833 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover,
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover *,
    [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
    [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover *,
    [data-testid="stSidebar"] [data-baseweb="select"] [role="option"]:hover,
    [data-testid="stSidebar"] [data-baseweb="select"] [role="option"]:hover * {
        background-color: rgba(0, 255, 133, 0.2) !important;
    }

    /* Group table (used for standings) */
    .table-wrapper {
        overflow-x: auto;
        border-radius: 12px;
        border: 1px solid #2c3e50;
        background-color: #0b0c10;
    }
    .group-table {
        width: 100%;
        border-collapse: collapse;
        font-variant-numeric: tabular-nums;
    }
    .group-table th {
        background-color: #37003c;
        color: #00ff85;
        padding: 0.65rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 700;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #00ff85;
    }
    .group-table td {
        padding: 0.55rem 0.5rem;
        border-bottom: 1px solid #2c3e50;
        text-align: center;
        font-size: 0.85rem;
        color: #f5f6f7;
    }
    .group-table tr:nth-child(even) { background-color: #161b22; }
    .group-table tr:hover { background-color: rgba(0, 255, 133, 0.08); }
    .group-table td:first-child { color: #a0a0a0; }
    .qualified { color: #00ff85; font-weight: 700; }
    .relegation { color: #e74c3c; font-weight: 700; }
</style>
"""

_STANDINGS_TABLE_CSS = dedent(
    """
    <style>
        .table-wrapper {
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid #2c3e50;
            background-color: #0b0c10;
        }
        .group-table {
            width: 100%;
            border-collapse: collapse;
            font-variant-numeric: tabular-nums;
        }
        .group-table th {
            background-color: #37003c;
            color: #00ff85;
            padding: 0.65rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 700;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #00ff85;
        }
        .group-table td {
            padding: 0.55rem 0.5rem;
            border-bottom: 1px solid #2c3e50;
            text-align: center;
            font-size: 0.85rem;
            color: #f5f6f7;
        }
        .group-table tr:nth-child(even) { background-color: #161b22; }
        .group-table tr:hover { background-color: rgba(0, 255, 133, 0.08); }
        .group-table td:first-child { color: #a0a0a0; }
        .qualified { color: #00ff85; font-weight: 700; }
        .relegation { color: #e74c3c; font-weight: 700; }
    </style>
    """
).strip()


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
            # Rows of the matrix are home goals, and Plotly maps rows to y.
            hovertemplate="Home: %{y} goals<br>Away: %{x} goals<br>Prob: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Scoreline Probability Matrix (%)",
        xaxis_title="Away Goals",
        yaxis_title="Home Goals",
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )
    return fig


def create_standings_table(rows: List[Dict], zones: Optional[Dict[str, int]] = None) -> str:
    # Zones are per-league counts of top/bottom positions to highlight
    # (e.g. {"qualification": 4, "relegation": 3}). Leagues without a
    # defined zone scheme render uncoloured.
    size = len(rows)
    zones = zones or {}
    qual_cutoff = int(zones.get("qualification") or 0)
    rel_count = int(zones.get("relegation") or 0)
    rel_cutoff = size - rel_count + 1 if rel_count else None
    body = ""
    for row in rows:
        cls = ""
        try:
            pos = int(row.get("position", 99))
        except (ValueError, TypeError):
            pos = 99
        if qual_cutoff and pos <= qual_cutoff:
            cls = "qualified"
        elif rel_cutoff and pos >= rel_cutoff:
            cls = "relegation"
        try:
            gd = int(row.get("goal_difference", 0) or 0)
        except (ValueError, TypeError):
            gd = 0
        gd_str = f"{gd:+d}" if gd else "0"
        body += dedent(
            f"""
            <tr class="{cls}">
                <td>{pos}</td>
                <td class="{cls}" style="text-align:left; padding-left:0.5rem;">{safe_html(row.get('team', ''))}</td>
                <td>{row.get('played', 0)}</td>
                <td>{row.get('won', 0)}</td>
                <td>{row.get('drawn', 0)}</td>
                <td>{row.get('lost', 0)}</td>
                <td>{row.get('goals_for', 0)}</td>
                <td>{row.get('goals_against', 0)}</td>
                <td>{gd_str}</td>
                <td style="font-weight:700;">{row.get('points', 0)}</td>
            </tr>
            """
        ).strip()
    table_html = dedent(
        f"""
        <div class="table-wrapper">
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
        </div>
        """
    ).strip()
    return _STANDINGS_TABLE_CSS + "\n" + table_html


def match_card_html(
    home_team: str,
    away_team: str,
    home_score: Optional[int],
    away_score: Optional[int],
    status: str,
    minutes: str,
    round_label: str = "",
    time: str = "",
    venue: str = "",
    prediction: Optional[Dict] = None,
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

    extras = ""
    if venue:
        extras += dedent(
            f"""
            <div style="font-size:0.8rem; color:#a0a0a0; text-align:center; margin-top:0.25rem;">
                📍 {safe_html(venue)}
            </div>
            """
        ).strip()
    if prediction:
        expected = prediction.get("expected_goals") or {}
        extras += prediction_summary_html(prediction, home_team, away_team)
        extras += (
            f"<div style='font-size:0.75rem;color:#a0a0a0;margin-top:0.2rem;'>"
            f"Expected goals: {expected.get('home', 0):.1f} – {expected.get('away', 0):.1f}</div>"
        )

    return dedent(
        f"""
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
                    <div style="font-size:0.75rem; color:#a0a0a0;">{safe_html(time)}{" SAST" if ":" in str(time) else ""}</div>
                </div>
                <div style="flex:1; text-align:left; font-weight:700; font-size:1.1rem;">{safe_html(away_team)}</div>
            </div>
            {extras}
        </div>
        """
    ).strip()
