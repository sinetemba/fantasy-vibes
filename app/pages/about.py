"""About page with project information."""

import streamlit as st

from app.ui import apply_theme, display_disclaimer, display_header, league_name
from src.data import LeagueService, prediction_log

TRACK_RECORD_DAYS = 30


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

    _render_track_record()

    st.markdown("---")
    display_disclaimer()


def _render_track_record():
    """Live accuracy of logged pre-kickoff predictions — the model's own
    track record, not a backtest."""
    st.markdown("### 📏 Prediction track record")
    report = prediction_log.summary(days=TRACK_RECORD_DAYS)
    if not report["n"]:
        st.info(
            "Tracking is running: every fixture in the coming week gets a "
            "prediction logged before kickoff, and its accuracy appears here "
            "once results come in."
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Likeliest outcome correct",
        f"{report['hit_rate'] * 100:.0f}%",
        help="How often the outcome the model rated most likely actually happened.",
    )
    c2.metric(
        "Always picking the home side",
        f"{report['home_baseline'] * 100:.0f}%",
        help="The naive baseline the model should beat.",
    )
    c3.metric(
        "Brier score",
        f"{report['brier']:.3f}",
        help="Error of the full probability forecast — lower is better; "
        "0.667 is what blind one-third guesses score.",
    )
    leagues = " · ".join(
        f"{league_name(code)} {rate * 100:.0f}% ({n})"
        for code, (rate, n) in report["per_league"].items()
    )
    st.caption(
        f"{report['n']} fixtures settled in the last {report['days']} days, scored on the "
        f"last prediction made before kickoff. By league: {leagues}"
    )

