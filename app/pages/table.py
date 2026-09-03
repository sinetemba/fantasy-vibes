"""League table page."""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

    table_height = max(180, 80 + len(standings) * 50)
    components.html(create_standings_table(standings), height=table_height, scrolling=False)

    st.markdown("---")
    with st.expander("📋 Raw table data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    display_disclaimer()
