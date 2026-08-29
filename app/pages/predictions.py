"""Match predictions page with visual analytics."""

from datetime import datetime

import streamlit as st

from app.ui import (
    apply_theme,
    create_outcome_bars,
    create_outcome_pie_chart,
    create_scoreline_chart,
    display_disclaimer,
    display_header,
    probability_bar,
    show_error_message,
    COLORS,
)
from src.data import LeagueService


def render(service: LeagueService):
    apply_theme()
    display_header(service.get_current_name(), "Data-driven match predictions")

    st.markdown("## 🧠 Prediction Model")
    st.caption("The model is automatically retrained from the latest fixtures. You can retrain on demand below.")
    if st.button("🔄 Retrain Model", type="primary", use_container_width=True):
        with st.spinner("Retraining model from latest fixtures..."):
            status = service.train()
        if status.get("fitted"):
            st.success(
                f"Model retrained on {status['matches_used']} matches across {status['teams_in_model']} teams"
            )
        else:
            st.warning("Could not fit a model — not enough full-time match data.")

    st.markdown("## 🔮 Match Prediction")
    teams = service.get_teams()
    if len(teams) < 2:
        st.info("At least two teams are needed for predictions.")
        return

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("🏠 Home Team", teams, index=0, key="pred_home")
    with c2:
        away_options = [t for t in teams if t != home]
        away = st.selectbox("✈️ Away Team", away_options, index=0, key="pred_away")

    neutral = st.checkbox("Neutral venue (no home advantage)", value=False, key="pred_neutral")

    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        if home == away:
            show_error_message("Please select two different teams.")
            return
        pred = service.predict(home, away, neutral=neutral)
        st.session_state["last_prediction"] = pred

    if "last_prediction" in st.session_state:
        _display_prediction(st.session_state["last_prediction"], service)

    st.markdown("---")
    display_disclaimer()


def _display_prediction(pred, service: LeagueService):
    home = pred["home_team"]
    away = pred["away_team"]
    outcome = pred["outcome_probabilities"]
    expected = pred["expected_goals"]
    score = pred["predicted_score"]

    st.markdown("---")
    st.markdown(f"### 📊 {home} vs {away}")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown(f"### 🏠 {home}")
        stats = service.get_team_stats(home)
        if stats:
            st.caption(f"P {stats['played']} · Pts {stats['points']} · GD {stats['goal_difference']:+d}")
    with c2:
        st.markdown(
            f"""
            <div style="text-align:center; padding:1rem;">
                <div style="font-size:0.9rem; color:#a0a0a0;">Predicted Score</div>
                <div style="font-size:2.5rem; font-weight:800; color:#00ff85;">{score['score']}</div>
                <div style="font-size:0.8rem; color:#a0a0a0;">{score['probability']*100:.1f}% likelihood · {pred['model']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(f"### ✈️ {away}")
        stats = service.get_team_stats(away)
        if stats:
            st.caption(f"P {stats['played']} · Pts {stats['points']} · GD {stats['goal_difference']:+d}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"⚽ {home} xG", f"{expected['home']:.2f}")
    with c2:
        st.metric("⚽ Total xG", f"{expected['home'] + expected['away']:.2f}")
    with c3:
        st.metric(f"⚽ {away} xG", f"{expected['away']:.2f}")

    st.markdown("---")
    st.markdown("### 📊 Outcome Probabilities")
    cb, cp = st.columns([3, 2])
    with cb:
        probability_bar(f"🏠 {home}", outcome["home_win"], COLORS["home_win"])
        probability_bar("🤝 Draw", outcome["draw"], COLORS["draw"])
        probability_bar(f"✈️ {away}", outcome["away_win"], COLORS["away_win"])
        st.plotly_chart(create_outcome_bars(outcome["home_win"], outcome["draw"], outcome["away_win"]), use_container_width=True)
    with cp:
        st.plotly_chart(create_outcome_pie_chart(outcome["home_win"], outcome["draw"], outcome["away_win"]), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Markets")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        ou = pred.get("over_under_2_5", {})
        st.metric("Over 2.5", f"{ou.get('over', 0)*100:.1f}%")
    with m2:
        st.metric("Under 2.5", f"{ou.get('under', 0)*100:.1f}%")
    with m3:
        btts = pred.get("btts", {})
        st.metric("BTTS Yes", f"{btts.get('yes', 0)*100:.1f}%")
    with m4:
        st.metric("BTTS No", f"{btts.get('no', 0)*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🎯 Scoreline Probability Matrix")
    matrix = pred.get("score_matrix")
    if matrix is not None:
        st.plotly_chart(create_scoreline_chart(matrix, max_goals=5), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Attack / Defense Ratings")
    a1, a2 = st.columns(2)
    with a1:
        st.metric(f"{home} attack", f"{pred['team_attack_params']['home']:.3f}")
        st.metric(f"{home} defense", f"{pred['team_defense_params']['home']:.3f}")
    with a2:
        st.metric(f"{away} attack", f"{pred['team_attack_params']['away']:.3f}")
        st.metric(f"{away} defense", f"{pred['team_defense_params']['away']:.3f}")

    st.caption(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
