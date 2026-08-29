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
        st.markdown(
            match_card_html(
                next_match["home_team"],
                next_match["away_team"],
                next_match.get("home_score"),
                next_match.get("away_score"),
                next_match.get("status", "scheduled"),
                next_match.get("minutes_elapsed", "TBD"),
                round_label=next_match.get("round", ""),
                time=next_match.get("date", "")[11:16] if next_match.get("date") else "",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    display_disclaimer()
    st.caption(f"{service.get_current_name()} · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
