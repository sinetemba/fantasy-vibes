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
    zone_legend_html,
)
from src.data import LeagueService


def render(service: LeagueService):
    apply_theme()
    display_header(service.get_current_name(), "League standings")

    st.markdown("## 📊 League Table")
    standings = service.get_standings()
    if not standings:
        show_info_message("No standings available for this league. Try another league from the sidebar.")
        return

    table_height = max(180, 80 + len(standings) * 50)
    zones = service.get_zones()
    components.html(
        create_standings_table(standings, zones=zones),
        height=table_height,
        scrolling=False,
    )
    legend = zone_legend_html(zones)
    if legend:
        st.markdown(legend, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📋 Raw table data"):
        cols = [
            "position", "team", "played", "won", "drawn", "lost",
            "goals_for", "goals_against", "goal_difference", "points",
        ]
        df = pd.DataFrame(standings)
        # Sources don't all return every column — select what exists.
        df = df[[c for c in cols if c in df.columns]]
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    display_disclaimer()
