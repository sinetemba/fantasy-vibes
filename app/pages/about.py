"""About page with project information."""

import streamlit as st

from app.ui import apply_theme, display_disclaimer, display_header
from src.data import LeagueService


def render(service: LeagueService):
    apply_theme()
    display_header("About", "Multi-league football data & predictions")

    st.markdown(r"""
    ### 🏆 Fantasy Vibes

    A Streamlit app for exploring football league data, fixtures, standings and match
    predictions. It is built around a flexible, multi-source data layer so that new
    leagues and providers can be added with minimal changes.

    ### ✨ Features
    - **Multi-league support** — switch between the Premier League, Bundesliga, Serie A,
      La Liga, Ligue 1 and more.
    - **Multiple free data sources** — cascades across `football-data.org`, `thesportsdb.com`
      and the public `openfootball/football.json` dataset.
    - **Night mode UI** — dark theme by default with custom cards and visualisations.
    - **Predictions** — attack/defense strength + Poisson model, with a neutral-venue option.
    - **Fixtures & table** — filter by round and date, predict individual fixtures.
    - **Stats** — team form, head-to-head records and league trend charts.

    ### 🧱 Data Sources
    - **Football-Data.org** — real current-season matches and tables (free API key required).
    - **TheSportsDB** — tables and fixtures (free key; demo key `3` included for testing).
    - **Openfootball** — public-domain match schedules and results, no API key needed.
    - **FixtureDownload.com** — CSV/JSON fixtures and results for many leagues.
    - **PSL Match Centre** — South African PSL fixtures and results via the official match centre.

    ### ⚙️ Getting Started
    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app/main.py
    ```
    """
    )

    if service is not None:
        sources = service.get_data_sources()
        st.markdown("### 📡 Active data sources")
        st.caption(
            f"Current league ({service.get_current_name()}): matches from "
            f"**{sources.get('matches') or 'none'}**, table from "
            f"**{sources.get('standings') or 'none'}**."
        )

    st.markdown("### 📏 Track record")
    st.caption(
        "Every prediction is logged before kickoff and scored once the result "
        "is in — see the **Track Record** page for accuracy, confidence levels, "
        "goals markets, the bookmaker comparison and the full history."
    )

    st.markdown("---")
    display_disclaimer()
