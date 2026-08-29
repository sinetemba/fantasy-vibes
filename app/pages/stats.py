"""Statistics page: team form, head-to-head and league trends."""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import (
    apply_theme,
    display_disclaimer,
    display_header,
    show_info_message,
)
from src.data import LeagueService


def render(service: LeagueService):
    apply_theme()
    display_header(service.get_current_name(), "Team form & league stats")

    tab1, tab2, tab3 = st.tabs(["📈 Team Form", "🤝 Head-to-Head", "⚽ League Trends"])

    with tab1:
        _render_form(service)
    with tab2:
        _render_h2h(service)
    with tab3:
        _render_trends(service)

    st.markdown("---")
    display_disclaimer()


def _render_form(service: LeagueService):
    st.markdown("### 📈 Recent Form")
    teams = service.get_teams()
    if not teams:
        show_info_message("No team data available.")
        return

    team = st.selectbox("Select a team", teams, key="form_team")
    n = st.slider("Number of matches", 3, 10, 5, key="form_n")
    form = service.get_team_form(team, n=n)
    if not form:
        st.info("No completed matches for this team yet.")
        return

    for f in form:
        color = {"W": "#00ff85", "D": "#f1c40f", "L": "#e74c3c"}.get(f["result"], "#a0a0a0")
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:0.5rem; background:#1f2833; border-radius:8px; margin:0.3rem 0;">
                <span style="font-weight:700; color:{color};">{f['result']}</span>
                <span style="color:#f5f6f7;">{f['venue']} · {f['opponent']}</span>
                <span style="color:#a0a0a0;">{f['score']}</span>
                <span style="color:#a0a0a0; font-size:0.75rem;">{f['date'][:10] if f['date'] else ''}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_h2h(service: LeagueService):
    st.markdown("### 🤝 Head-to-Head")
    teams = service.get_teams()
    if len(teams) < 2:
        show_info_message("Need at least two teams.")
        return

    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Team A", teams, key="h2h_a")
    with c2:
        b = st.selectbox("Team B", [t for t in teams if t != a], key="h2h_b")

    record, matches = service.get_h2h(a, b)
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-around; padding:1rem; background:#1f2833; border-radius:12px; margin:1rem 0;">
            <div style="text-align:center;"><div style="font-size:1.5rem; font-weight:800; color:#00ff85;">{record['a_wins']}</div><div>{a} wins</div></div>
            <div style="text-align:center;"><div style="font-size:1.5rem; font-weight:800; color:#f1c40f;">{record['draws']}</div><div>Draws</div></div>
            <div style="text-align:center;"><div style="font-size:1.5rem; font-weight:800; color:#e74c3c;">{record['b_wins']}</div><div>{b} wins</div></div>
            <div style="text-align:center;"><div style="font-size:1.5rem; font-weight:800;">{record['a_gf']}:{record['a_ga']}</div><div>Goals ({a}:{b})</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if matches:
        st.markdown("#### Recent meetings")
        for m in list(reversed(matches))[:5]:
            st.markdown(f"{m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']} · {m['date'][:10] if m['date'] else ''}")


def _render_trends(service: LeagueService):
    st.markdown("### ⚽ League Trends")
    standings = service.get_standings()
    if not standings:
        show_info_message("No standings to plot.")
        return

    df = pd.DataFrame(standings)
    fig = px.scatter(
        df,
        x="goals_for",
        y="points",
        text="team",
        title="Goals For vs Points",
        labels={"goals_for": "Goals For", "points": "Points"},
        color_discrete_sequence=["#00ff85"],
    )
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f5f6f7"),
        xaxis=dict(gridcolor="#2c3e50"),
        yaxis=dict(gridcolor="#2c3e50"),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
