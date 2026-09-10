"""Today's fixtures page showing matches across all configured leagues."""

import logging
from datetime import date, datetime
from itertools import groupby
from typing import Any, Dict, List

import streamlit as st

from app.ui import (
    apply_theme,
    display_disclaimer,
    display_header,
    match_card_html,
)
from src.data import LeagueService

logger = logging.getLogger(__name__)


def _match_date(m: Dict[str, Any]) -> date:
    try:
        return datetime.fromisoformat(m["date"]).date()
    except Exception:
        return date.min


@st.cache_data(ttl=300, show_spinner=False)
def _load_todays_fixtures(today: str) -> List[Dict[str, Any]]:
    target = date.fromisoformat(today)
    fixtures: List[Dict[str, Any]] = []
    base = LeagueService()
    for code, name in base.get_leagues():
        try:
            league_service = LeagueService(code)
        except Exception as exc:
            logger.error(f"Could not load league {code}: {exc}")
            continue
        for m in league_service.get_matches():
            if _match_date(m) != target:
                continue
            row = dict(m)
            row["league_code"] = code
            row["league_name"] = name
            fixtures.append(row)
    fixtures.sort(key=lambda m: (m["league_name"], m.get("date") or ""))
    return fixtures


def _clear_cache():
    _load_todays_fixtures.clear()


def render(service: LeagueService):
    apply_theme()

    today = date.today()
    with st.spinner("Loading today's fixtures..."):
        matches = _load_todays_fixtures(today.isoformat())

    today_str = today.strftime("%A, %d %B %Y")
    if matches:
        leagues = sorted({m["league_name"] for m in matches})
        subtitle = f"{', '.join(leagues)} - {today_str}"
    else:
        subtitle = f"All leagues - {today_str}"
    display_header("Today's Fixtures", subtitle)

    if not matches:
        return

    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown("Fixture list aggregated across all configured leagues for the current day.")
    with c2:
        if st.button("🔄 Refresh", use_container_width=True):
            _clear_cache()
            st.rerun()

    live_count = sum(1 for m in matches if m.get("status") == "live")
    done_count = sum(1 for m in matches if m.get("status") == "full_time")
    scheduled_count = sum(1 for m in matches if m.get("status") == "scheduled")

    cols = st.columns(3)
    cols[0].metric("Live", live_count)
    cols[1].metric("Completed", done_count)
    cols[2].metric("Scheduled", scheduled_count)

    for league_name, league_matches in groupby(matches, key=lambda m: m["league_name"]):
        league_list = list(league_matches)
        st.markdown(f"### 🏆 {league_name} ({len(league_list)})")
        for m in league_list:
            _render_match(m)

    display_disclaimer()


def _render_match(m: Dict[str, Any]):
    time = ""
    if m.get("date"):
        try:
            dt = datetime.fromisoformat(m["date"])
            time = dt.strftime("%H:%M")
        except Exception:
            pass

    minutes = m.get("minutes_elapsed", "TBD")
    if m.get("status") == "full_time":
        minutes = "FT"
    elif m.get("status") == "live":
        minutes = minutes or "LIVE"

    st.markdown(
        match_card_html(
            m["home_team"],
            m["away_team"],
            m.get("home_score"),
            m.get("away_score"),
            m.get("status", "scheduled"),
            str(minutes),
            round_label=m.get("round", ""),
            time=time,
        ),
        unsafe_allow_html=True,
    )
