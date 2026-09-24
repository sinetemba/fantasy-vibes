"""Live matches page showing live, recent and upcoming fixtures."""

from datetime import datetime, timedelta

import streamlit as st

from app.ui import (
    apply_theme,
    display_disclaimer,
    display_header,
    display_metric_card,
    match_card_html,
    show_info_message,
    show_warning_message,
)
from src.data import LeagueService
from src.data.utils import is_upcoming


def _refresh(service: LeagueService):
    """Public refresh helper for the current league."""
    service.refresh()


def render(service: LeagueService):
    apply_theme()
    display_header(service.get_current_name(), "Live, recent and upcoming matches")

    st.markdown("## ⚽ Live & Upcoming")

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown("Real-time and recent match data pulled from the active data source.")
    with c2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            _refresh(service)
            st.rerun()
    with c3:
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="live_auto_refresh")

    interval = None
    if auto_refresh:
        interval = st.slider("Refresh interval (s)", 30, 300, 60, key="live_refresh_interval")

    # Auto-refresh reruns just this fragment rather than reloading the whole
    # page — same freshness, much lighter.
    if hasattr(st, "fragment"):
        run_every = timedelta(seconds=interval) if interval else None

        @st.fragment(run_every=run_every)
        def _live_body():
            _render_match_lists(service)

        _live_body()
    else:
        _render_match_lists(service)

    display_disclaimer()


def _render_match_lists(service: LeagueService):
    matches = service.get_matches()
    if not matches:
        show_warning_message("No match data available for this league.")
        return

    live = service.get_live_matches()
    full_time = [m for m in matches if m.get("status") == "full_time"]
    scheduled = [m for m in matches if is_upcoming(m)]

    if live:
        st.markdown("### 🔴 Live Now")
        for m in live:
            _render_match(m)

    if full_time:
        st.markdown("### ✅ Recent Results")
        for m in list(reversed(full_time))[:10]:
            _render_match(m)

    if scheduled:
        st.markdown(f"### 📅 Upcoming ({len(scheduled)})")
        for m in scheduled[:10]:
            _render_match(m)

    st.markdown("---")
    st.markdown("### 📈 Match Stats")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        display_metric_card("Live", str(len(live)), "")
    with c2:
        display_metric_card("Completed", str(len(full_time)), "")
    with c3:
        display_metric_card("Upcoming", str(len(scheduled)), "")
    with c4:
        total = sum(
            (m["home_score"] or 0) + (m["away_score"] or 0)
            for m in matches
            if m.get("status") == "full_time"
        )
        display_metric_card("Goals", str(total), "")

    if not live and not full_time and not scheduled:
        show_info_message("No matches found in any status filter.")

    st.markdown("---")


def _render_match(m):
    time = ""
    if m.get("date"):
        try:
            dt = datetime.fromisoformat(m["date"])
            time = dt.strftime("%H:%M")
        except Exception:
            pass
    if m.get("time_confirmed") is False:
        time = "TBC"
    st.markdown(
        match_card_html(
            m["home_team"],
            m["away_team"],
            m.get("home_score"),
            m.get("away_score"),
            m.get("status", "scheduled"),
            m.get("minutes_elapsed", "TBD"),
            round_label=m.get("round", ""),
            time=time,
        ),
        unsafe_allow_html=True,
    )
