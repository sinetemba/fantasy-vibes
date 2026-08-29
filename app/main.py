"""Main entry point for the Fantasy Vibes multi-league football app."""

import logging
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from app.ui import apply_theme
from src.data import LeagueService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("fantasy_vibes")

st.set_page_config(
    page_title="Fantasy Vibes",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "🏠 Home": "Home",
    "⚽ Live": "Live",
    "📅 Fixtures": "Fixtures",
    "📊 Table": "Table",
    "🔮 Predictions": "Predictions",
    "📈 Stats": "Stats",
    "ℹ️ About": "About",
}


def init_session_state():
    if "league_service" not in st.session_state:
        default = _default_league_code()
        st.session_state.league_service = LeagueService(default)
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "last_league" not in st.session_state:
        st.session_state.last_league = st.session_state.league_service.get_current_code()


def _default_league_code() -> str:
    try:
        service = LeagueService()
        for code, _ in service.get_leagues():
            if service._leagues.get(code, {}).get("default"):
                return code
        return service.get_leagues()[0][0]
    except Exception:
        return "PL"


def _change_league():
    new_code = st.session_state.get("league_select")
    if new_code and new_code != st.session_state.league_service.get_current_code():
        st.session_state.league_service.set_league(new_code)
        st.session_state.last_league = new_code
        st.rerun()


def render_sidebar():
    service: LeagueService = st.session_state.league_service

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding:1.2rem 0.5rem 0.5rem;">
                <div style="font-size:2.5rem; line-height:1;">🏆</div>
                <div style="font-size:1.2rem; font-weight:800; color:#00ff85; margin-top:0.3rem;">
                    Fantasy Vibes
                </div>
                <div style="font-size:0.75rem; color:#a0a0a0;">
                    Multi-league football data
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # League selector
        leagues = service.get_leagues()
        default_index = [c for c, _ in leagues].index(service.get_current_code())
        st.selectbox(
            "🏆 Select League",
            options=[c for c, _ in leagues],
            format_func=lambda x: service._leagues[x]["name"],
            index=default_index,
            key="league_select",
            on_change=_change_league,
        )

        st.markdown("---")

        # Navigation
        selected = None
        for label, page_name in PAGES.items():
            if st.button(
                label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if st.session_state.current_page == page_name else "secondary",
            ):
                selected = page_name
        if selected:
            st.session_state.current_page = selected
            st.rerun()

        st.markdown("---")

        # Data source status
        sources = service.get_data_sources()
        st.markdown("##### 📡 Data Sources")
        for key, val in sources.items():
            color = "#00ff85" if val else "#e74c3c"
            label = val or "none"
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin:0.2rem 0;">
                    <span style="color:#a0a0a0;">{key.title()}</span>
                    <span style="color:{color}; font-weight:700;">{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(
            "v1.0.0 · Built with Streamlit · Data from public football datasets"
        )


def render_page():
    service: LeagueService = st.session_state.league_service
    page = st.session_state.current_page

    try:
        if page == "Home":
            from app.pages import home
            home.render(service)
        elif page == "Live":
            from app.pages import live
            live.render(service)
        elif page == "Fixtures":
            from app.pages import fixtures
            fixtures.render(service)
        elif page == "Table":
            from app.pages import table
            table.render(service)
        elif page == "Predictions":
            from app.pages import predictions
            predictions.render(service)
        elif page == "Stats":
            from app.pages import stats
            stats.render(service)
        elif page == "About":
            from app.pages import about
            about.render(service)
    except Exception as e:
        logger.error(f"Error rendering {page}: {e}", exc_info=True)
        st.error(f"Could not load the {page} page. Please refresh.")


def main():
    apply_theme()
    init_session_state()
    render_sidebar()
    render_page()


if __name__ == "__main__":
    main()
