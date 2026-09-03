"""Home page overview with quick stats and a prediction teaser."""

from datetime import datetime

import streamlit as st

from app.ui import (
    apply_theme,
    display_disclaimer,
    display_header,
    display_metric_card,
    match_card_html,
    show_info_message,
)
from src.data import LeagueService


def _render_favourite_team(service: LeagueService, team: str):
    st.markdown("### ⭐ Your Favourite Team")
    stats = service.get_team_stats(team)
    if stats:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Position", stats.get("position", "—"))
        with c2:
            st.metric("Points", stats.get("points", "—"))
        with c3:
            st.metric("Form", stats.get("form", "—"))
    else:
        st.caption("Team data not available.")

    with st.expander("Recent form"):
        form = service.get_team_form(team, n=5)
        if form:
            for m in form:
                date = m.get("date", "")[:10]
                st.caption(f"{date} {m.get('venue', '')} vs {m.get('opponent', '')} {m.get('score', '')} ({m.get('result', '')})")
        else:
            st.caption("No recent form data.")

    upcoming = [
        m
        for m in service.get_matches()
        if m.get("status") == "scheduled" and (m["home_team"] == team or m["away_team"] == team)
    ]
    if upcoming:
        upcoming.sort(key=lambda m: m.get("date") or "z")
        next_match = upcoming[0]
        st.markdown("**Next match**")
        try:
            dt = datetime.fromisoformat(next_match.get("date")) if next_match.get("date") else None
            kickoff = dt.strftime("%a, %d %b %Y %H:%M") if dt else ""
        except Exception:
            kickoff = ""
        st.markdown(
            match_card_html(
                next_match["home_team"],
                next_match["away_team"],
                next_match.get("home_score"),
                next_match.get("away_score"),
                next_match.get("status", "scheduled"),
                next_match.get("minutes_elapsed", "TBD"),
                round_label=next_match.get("round", ""),
                time=kickoff,
            ),
            unsafe_allow_html=True,
        )
        if st.button("🔮 Predict this match", use_container_width=True, key="fav_predict"):
            st.session_state["pred_home"] = next_match["home_team"]
            st.session_state["pred_away"] = next_match["away_team"]
            st.session_state.current_page = "Predictions"
            st.rerun()
    else:
        st.info("No upcoming fixtures for this team.")


def render(service: LeagueService):
    apply_theme()
    display_header(service.get_current_name(), "Live data, standings & match predictions")

    st.markdown("---")

    standings = service.get_standings()
    matches = service.get_matches()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f"""
            ### 🏆 Welcome to the {service.get_current_name()} Hub

            Explore fixtures, live scores, the league table, detailed stats and
            data-driven match predictions. The app pulls from multiple free data
            sources and is designed so new leagues can be added easily.
            """
        )
        sources = service.get_data_sources()
        if sources.get("standings"):
            show_info_message(f"Standings loaded from **{sources['standings']}**")
        else:
            st.warning("No live data source currently available. Check your API keys or .env file.")

    with col2:
        if standings:
            leader = standings[0]
            display_metric_card("League Leader", leader.get("team", "—"), f"{leader.get('points', 0)} pts")
        else:
            display_metric_card("League Leader", "—", "")
        display_metric_card("Matches Loaded", f"{len(matches)}", "")

    st.markdown("---")

    # Favourite team watchlist
    favourite = st.session_state.get("favourite_team")
    if favourite and favourite in service.get_teams():
        _render_favourite_team(service, favourite)
        st.markdown("---")

    # Top teams mini-table
    if standings:
        st.markdown("### 🥇 Top Teams")
        cols = st.columns(3)
        for i, row in enumerate(standings[:3]):
            with cols[i]:
                icon = ["🥇", "🥈", "🥉"][i]
                display_metric_card(
                    f"{icon} {row.get('team', '')}",
                    f"{row.get('points', 0)} pts",
                    f"W {row.get('won', 0)} · D {row.get('drawn', 0)} · L {row.get('lost', 0)}",
                )

    # Quick prediction
    st.markdown("---")
    st.markdown("### 🔮 Quick Prediction")
    teams = service.get_teams()
    if len(teams) >= 2:
        col_h, col_a = st.columns(2)
        with col_h:
            home = st.selectbox("Home Team", teams, index=0, key="home_quick")
        with col_a:
            away = st.selectbox("Away Team", [t for t in teams if t != home], index=0, key="away_quick")

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            pred = service.predict(home, away)
            st.session_state["last_prediction"] = pred
            st.session_state["prediction_home"] = home
            st.session_state["prediction_away"] = away

        if "last_prediction" in st.session_state:
            pred = st.session_state["last_prediction"]
            outcome = pred.get("outcome_probabilities", {})
            c1, c2, c3 = st.columns(3)
            c1.metric(f"🏠 {home}", f"{outcome.get('home_win', 0)*100:.1f}%")
            c2.metric("🤝 Draw", f"{outcome.get('draw', 0)*100:.1f}%", pred["predicted_score"]["score"])
            c3.metric(f"✈️ {away}", f"{outcome.get('away_win', 0)*100:.1f}%")

    # Featured next fixture
    st.markdown("---")
    upcoming = [m for m in matches if m.get("status") == "scheduled"]
    if upcoming:
        st.markdown("### 📅 Next Fixture")
        next_match = upcoming[0]
        try:
            dt = datetime.fromisoformat(next_match.get("date")) if next_match.get("date") else None
            fixture_time = dt.strftime("%a, %d %b %Y %H:%M") if dt else ""
        except Exception:
            fixture_time = ""
        st.markdown(
            match_card_html(
                next_match["home_team"],
                next_match["away_team"],
                next_match.get("home_score"),
                next_match.get("away_score"),
                next_match.get("status", "scheduled"),
                next_match.get("minutes_elapsed", "TBD"),
                round_label=next_match.get("round", ""),
                time=fixture_time,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    display_disclaimer()
    st.caption(f"{service.get_current_name()} · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
