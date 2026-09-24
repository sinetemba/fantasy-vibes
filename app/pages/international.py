"""International page — friendlies and qualifiers across confederations."""

import logging
from datetime import date, datetime, timedelta
from itertools import groupby
from typing import Any, Dict, List

import streamlit as st

from app.pages.fixtures import _render_prediction_summary
from app.ui import (
    apply_theme,
    display_disclaimer,
    display_header,
    get_league_service,
    prewarm_league_services,
    show_warning_message,
)
from src.data import LeagueService

GROUP = "international"
MAX_RENDERED = 150

logger = logging.getLogger(__name__)


def _parse_date(m: Dict[str, Any]) -> date:
    try:
        return datetime.fromisoformat(m["date"]).date()
    except Exception:
        return datetime.min.date()


def _date_label(m: Dict[str, Any]) -> str:
    try:
        return datetime.fromisoformat(m["date"]).strftime("%a, %d %b %Y")
    except Exception:
        return "TBD"


def _service_for(code: str) -> LeagueService:
    """Per-competition service for predictions, shared via the app cache."""
    return get_league_service(code)


def render(service: LeagueService):
    apply_theme()
    display_header("International", "Friendlies & qualifiers across confederations")

    st.markdown("## 🌍 International Fixtures")

    # Filters render immediately — the competition list comes from the
    # league registry, not the fetched data.
    comp_names = sorted(name for _, name in service.get_group_leagues(GROUP))
    _render_filter_widgets(comp_names)

    progress = st.progress(0, text="Loading competitions…")
    total = max(len(comp_names), 1)
    done = {"n": 0}

    def _on_progress(_code, name):
        done["n"] += 1
        progress.progress(done["n"] / total, text=f"Loaded {name}")

    matches = service.get_group_matches(GROUP, on_progress=_on_progress)
    progress.empty()

    if not matches:
        show_warning_message("No international fixtures available.")
        return

    if "intl_predictions" not in st.session_state:
        st.session_state["intl_predictions"] = {}

    filtered = _apply_filters(matches)

    # Warm prediction services for the competitions in view so the first
    # Predict click doesn't block on data fetch + model training.
    warmed = st.session_state.setdefault("intl_prewarmed", set())
    codes = []
    for m in filtered[:40]:
        code = m.get("competition")
        if code and code not in warmed and code not in codes:
            codes.append(code)
    warmed.update(codes)
    prewarm_league_services(codes)

    if not filtered:
        st.info("No matches fit the selected filters.")
        return

    c_head, c_btn = st.columns([4, 1])
    with c_head:
        pending = sum(
            1
            for m in filtered
            if m.get("status") != "full_time"
            and m["match_id"] not in st.session_state["intl_predictions"]
        )
        st.caption(f"{len(filtered)} matches · {pending} awaiting prediction")
    with c_btn:
        if st.button("🔮 Predict All", use_container_width=True, key="intl_predict_all"):
            _predict_all(filtered)
            st.rerun()

    for date_label, day_matches in groupby(filtered[:MAX_RENDERED], key=_date_label):
        st.markdown(f"### 📆 {date_label}")
        for m in day_matches:
            _render_match(m)
        st.markdown("---")

    if len(filtered) > MAX_RENDERED:
        st.caption(f"Showing the first {MAX_RENDERED} of {len(filtered)} matches — refine the filters to see more.")

    display_disclaimer()


def _render_filter_widgets(competitions: List[str]) -> None:
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.selectbox("🏆 Competition", ["All competitions"] + competitions, key="intl_competition")
    with c2:
        st.selectbox("📋 Show", ["Upcoming", "Results", "All"], key="intl_status")
    with c3:
        st.text_input("🔍 Team", placeholder="e.g. South Africa", key="intl_team")

    c4, c5 = st.columns([1, 1])
    with c4:
        st.checkbox("🗓️ This week only", value=False, key="intl_this_week")
    with c5:
        st.checkbox("📅 Filter by date", value=False, key="intl_use_date")
        if st.session_state.get("intl_use_date"):
            st.date_input(
                "Date",
                value=date.today(),
                key="intl_date",
                label_visibility="collapsed",
            )


def _apply_filters(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    competition = st.session_state.get("intl_competition", "All competitions")
    status = st.session_state.get("intl_status", "Upcoming")
    team_query = (st.session_state.get("intl_team") or "").strip().lower()
    this_week = st.session_state.get("intl_this_week", False)
    use_date = st.session_state.get("intl_use_date", False)
    date_filter = st.session_state.get("intl_date")

    filtered = matches
    if competition != "All competitions":
        filtered = [m for m in filtered if m.get("competition_name") == competition]
    if team_query:
        filtered = [
            m
            for m in filtered
            if team_query in m.get("home_team", "").lower()
            or team_query in m.get("away_team", "").lower()
        ]

    if this_week:
        start = date.today()
        end = start + timedelta(days=6)
        filtered = [
            m
            for m in filtered
            if m.get("date") and start <= _parse_date(m) <= end
        ]
        filtered.sort(key=lambda m: m.get("date") or "")
        return filtered

    if use_date and date_filter:
        try:
            d = date_filter if isinstance(date_filter, date) else date_filter.date()
            filtered = [m for m in filtered if _parse_date(m) == d]
        except Exception:
            pass

    # "Scheduled" matches dated in the past are stale feed entries with no
    # recorded score — don't present them as upcoming.
    today = date.today()
    upcoming = [
        m
        for m in filtered
        if m.get("status") == "scheduled" and _parse_date(m) >= today
    ]
    live = [m for m in filtered if m.get("status") in ("live", "half_time")]
    completed = [m for m in filtered if m.get("status") == "full_time"]
    other = [m for m in filtered if m.get("status") not in ("scheduled", "full_time", "live", "half_time")]

    upcoming.sort(key=lambda m: m.get("date") or "")
    completed.sort(key=lambda m: m.get("date") or "", reverse=True)

    if status == "Upcoming":
        return live + upcoming + other
    if status == "Results":
        return completed
    return live + upcoming + completed + other


def _predict_all(matches: List[Dict[str, Any]]) -> None:
    """Predict every displayed match that hasn't finished yet, using each
    match's own competition model."""
    predictions = st.session_state["intl_predictions"]
    pending = [
        m
        for m in matches
        if m.get("status") != "full_time"
        and m["match_id"] not in predictions
        and m.get("competition")
    ]
    if not pending:
        return

    progress = st.progress(0, text="Predicting…")
    total = len(pending)
    for i, m in enumerate(pending):
        try:
            pred = _service_for(m["competition"]).predict(
                m["home_team"], m["away_team"]
            )
            pred["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predictions[m["match_id"]] = pred
        except Exception as exc:
            logger.warning(f"Prediction failed for {m['match_id']}: {exc}")
        progress.progress(
            (i + 1) / total,
            text=f"Predicted {m['home_team']} v {m['away_team']}",
        )
    progress.empty()


def _render_match(m: Dict[str, Any]):
    home = m["home_team"]
    away = m["away_team"]
    fixture_id = m["match_id"]
    predictions = st.session_state["intl_predictions"]

    c_info, c_match, c_action = st.columns([1.4, 2.6, 1.2])

    with c_info:
        try:
            dt = datetime.fromisoformat(m["date"]) if m.get("date") else None
            time = dt.strftime("%H:%M") + " SAST" if dt else "—"
        except Exception:
            time = "—"
        if m.get("time_confirmed") is False:
            time = "TBC"
        st.markdown(
            f"""**{time}**  
<span style='color:#a0a0a0; font-size:0.8rem;'>{m.get('competition_name', '')}</span>""",
            unsafe_allow_html=True,
        )

    with c_match:
        if m.get("status") == "full_time" and m.get("home_score") is not None:
            st.markdown(f"{home} {m['home_score']}-{m['away_score']} {away}")
        else:
            st.markdown(f"{home} vs {away}")
        st.caption(f"Source: {m.get('source', 'unknown')}")

    with c_action:
        if m.get("status") == "full_time":
            st.caption("Completed")
        elif st.button("🔮 Predict", key=f"intl_predict_{fixture_id}", use_container_width=True):
            with st.spinner("Loading prediction model..."):
                comp_service = _service_for(m.get("competition"))
                pred = comp_service.predict(home, away)
            pred["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predictions[fixture_id] = pred
            st.rerun()

    if fixture_id in predictions:
        _render_prediction_summary(
            predictions[fixture_id], home, away, service=_service_for(m.get("competition"))
        )
