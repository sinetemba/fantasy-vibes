"""Match predictions page with visual analytics."""

from datetime import datetime

import streamlit as st

from app.ui import (
    apply_theme,
    create_scoreline_chart,
    display_disclaimer,
    display_header,
    form_badges_html,
    outcome_bar_html,
    prediction_caveat,
    prediction_reasons,
    prediction_verdict,
    safe_html,
    show_error_message,
    team_strength_lines,
    top_scorelines,
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

    # Pre-select teams if they were set elsewhere (e.g. favourite team on Home)
    if st.session_state.get("pred_home") not in teams:
        st.session_state["pred_home"] = teams[0]
    if st.session_state.get("pred_away") == st.session_state.get("pred_home"):
        st.session_state["pred_away"] = None

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("🏠 Home Team", teams, key="pred_home")
        _render_form(service.get_team_form(home))
    with c2:
        away_options = [t for t in teams if t != home]
        if st.session_state.get("pred_away") not in away_options:
            st.session_state["pred_away"] = away_options[0] if away_options else None
        away = st.selectbox("✈️ Away Team", away_options, key="pred_away")
        if away:
            _render_form(service.get_team_form(away))

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


def _render_form(form):
    """Render last-5 form as W/D/L badges (rightmost = most recent)."""
    st.markdown(
        f"<div style='margin-top:0.2rem'><span style='color:#a0a0a0;font-size:0.8rem;'>Form&nbsp;</span>"
        f"{form_badges_html(form)}"
        f"<span style='color:#a0a0a0;font-size:0.75rem;margin-left:4px;'>latest →</span></div>",
        unsafe_allow_html=True,
    )


def _display_prediction(pred, service: LeagueService):
    home = pred["home_team"]
    away = pred["away_team"]
    expected = pred["expected_goals"]

    st.markdown("---")
    st.markdown(f"### 📊 {home} vs {away}")

    # Verdict first — it's what people read — then the one outcome bar and
    # the reasons behind it.
    st.markdown(
        f"<div style='font-size:1.6rem;font-weight:800;color:#00ff85;margin:0.25rem 0;'>"
        f"{safe_html(prediction_verdict(pred))}</div>"
        f"{outcome_bar_html(pred)}",
        unsafe_allow_html=True,
    )
    reasons = prediction_reasons(pred)
    if reasons:
        st.markdown(reasons)
    caveat = prediction_caveat(pred)
    if caveat:
        st.warning(caveat)

    st.markdown("#### 🎯 Most likely scores")
    scores = top_scorelines(pred, n=3)
    score_cols = st.columns(len(scores) or 1)
    for col, (score, prob) in zip(score_cols, scores):
        col.metric(score, f"{prob * 100:.0f}%")
    st.caption(
        "Even the likeliest exact score happens only this often — "
        "the outcome bar above is the better guide."
    )

    st.markdown("#### 📋 Goals")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(f"{home} expected goals", f"{expected.get('home', 0):.1f}")
    g2.metric(f"{away} expected goals", f"{expected.get('away', 0):.1f}")
    g3.metric("Over 2.5 goals", f"{(pred.get('over_under_2_5') or {}).get('over', 0) * 100:.0f}%")
    g4.metric("Both teams score", f"{(pred.get('btts') or {}).get('yes', 0) * 100:.0f}%")

    strength = team_strength_lines(pred)
    home_stats = service.get_team_stats(home)
    away_stats = service.get_team_stats(away)
    if strength or home_stats or away_stats:
        st.markdown("#### 📈 Team strength")
        for line in strength:
            st.markdown(line)
        for team, stats in ((home, home_stats), (away, away_stats)):
            if stats:
                st.caption(f"{team}: {_stats_line(stats)}")

    matrix = pred.get("score_matrix")
    if matrix is not None:
        with st.expander("Full scoreline probabilities"):
            st.plotly_chart(create_scoreline_chart(matrix, max_goals=5), use_container_width=True)

    st.caption(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _stats_line(stats) -> str:
    """One-line P/Pts/GD summary that tolerates partial standings rows."""
    played = stats.get("played", "—")
    points = stats.get("points", "—")
    gd = stats.get("goal_difference")
    gd_txt = f"{gd:+d}" if isinstance(gd, int) else "—"
    return f"P {played} · Pts {points} · GD {gd_txt}"
