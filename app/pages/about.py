"""About page with project information."""

import streamlit as st

from app.ui import apply_theme, display_disclaimer, display_header, league_name
from src.data import LeagueService, odds, prediction_log

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
    _render_market_benchmark()


def _render_market_benchmark():
    """Model vs bookmaker odds on the same settled fixtures — the honest
    yardstick. Pre-match odds (market average, football-data.co.uk) are
    the strongest public forecast; the gap shows how far the model is
    from it."""
    try:
        bm = odds.benchmark(days=TRACK_RECORD_DAYS)
    except Exception:
        return
    if not bm.get("n"):
        return

    st.markdown("#### 🆚 Model vs the betting market")
    c1, c2 = st.columns(2)
    c1.metric(
        "Log-loss — this model",
        f"{bm['model_logloss']:.3f}",
        delta=f"{bm['model_logloss'] - bm['market_logloss']:+.3f} vs market",
        delta_color="inverse",
        help="Lower is better. Model probabilities before any market blend "
        "— the honest comparison. Positive delta means the market priced "
        "these fixtures better than the model did.",
    )
    c2.metric(
        "Log-loss — bookmaker odds",
        f"{bm['market_logloss']:.3f}",
        help="Implied probabilities from pre-match market-average odds "
        "(football-data.co.uk), margin removed.",
    )
    st.caption(
        f"Scored on {bm['n']} of {bm['total']} settled fixtures with "
        f"bookmaker odds available. Model hit rate {bm['model_hit'] * 100:.0f}% "
        f"vs market favourite {bm['market_hit'] * 100:.0f}%."
    )
    if bm.get("blend_n"):
        diff = bm["blend_base_logloss"] - bm["blend_logloss"]
        direction = "improved" if diff > 0 else "worsened"
        st.caption(
            f"Market odds were blended into {bm['blend_n']} of these predictions: "
            f"log-loss {bm['blend_base_logloss']:.3f} unblended → "
            f"{bm['blend_logloss']:.3f} blended ({direction} by {abs(diff):.3f})."
        )

