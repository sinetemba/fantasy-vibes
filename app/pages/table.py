"""League table page."""

import pandas as pd
import streamlit as st

from app.ui import (
    apply_theme,
    create_standings_table,
    display_disclaimer,
    display_header,
    show_info_message,
)
from src.data import LeagueService


def render(service: LeagueService):
    apply_theme()
    display_header(service.get_current_name(), "League standings")

    st.markdown("## 📊 League Table")
    standings = service.get_standings()
    if not standings:
        show_info_message("No standings available. Try another data source or season.")
        return

    df = pd.DataFrame(standings)
    df = df[
        ["position", "team", "played", "won", "drawn", "lost", "goals_for", "goals_against", "goal_difference", "points"]
    ]

    st.markdown(create_standings_table(standings), unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📋 Raw table data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    display_disclaimer()
