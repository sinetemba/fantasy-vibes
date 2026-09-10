"""Main entry point for the Fantasy Vibes multi-league football app."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREFERENCES_PATH = PROJECT_ROOT / "data" / "cache" / "preferences.json"
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
    "🗓️ Today's Fixtures": "TodaysFixtures",
    "📊 Table": "Table",
    "🔮 Predictions": "Predictions",
    "📈 Stats": "Stats",
    "ℹ️ About": "About",
}


def _load_preferences() -> Dict[str, Any]:
    try:
        with open(PREFERENCES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_preferences():
    try:
        PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_league": st.session_state.get("last_league"),
                    "favourite_team": st.session_state.get("favourite_team"),
                },
                f,
                indent=2,
            )
    except Exception:
        pass


def init_session_state():
    if "league_service" not in st.session_state:
        default = _default_league_code()
        st.session_state.league_service = LeagueService(default)
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "last_league" not in st.session_state:
        st.session_state.last_league = st.session_state.league_service.get_current_code()
    if "favourite_team" not in st.session_state:
        prefs = _load_preferences()
        teams = st.session_state.league_service.get_teams()
        fav = st.query_params.get("fav") or prefs.get("favourite_team")
        st.session_state.favourite_team = fav if fav in teams else (teams[0] if teams else None)


def _default_league_code() -> str:
    prefs = _load_preferences()
    candidate = st.query_params.get("league") or prefs.get("last_league")
    try:
        service = LeagueService()
        codes = [c for c, _ in service.get_leagues()]
        if candidate and candidate in codes:
            return candidate
        for code, _ in service.get_leagues():
            if service._leagues.get(code, {}).get("default"):
                return code
        return codes[0]
    except Exception:
        return candidate or "PL"


def _change_league():
    new_code = st.session_state.get("league_select")
    if new_code and new_code != st.session_state.league_service.get_current_code():
        st.session_state.league_service.set_league(new_code)
        st.session_state.last_league = new_code
        st.query_params["league"] = new_code
        _save_preferences()
        st.rerun()


def _change_favourite():
    new_fav = st.session_state.get("fav_select")
    if new_fav:
        st.session_state.favourite_team = new_fav
        st.query_params["fav"] = new_fav
        _save_preferences()


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

        # Favourite team
        teams = service.get_teams()
        if teams:
            fav = st.session_state.get("favourite_team")
            if not fav or fav not in teams:
                fav = teams[0]
                st.session_state.favourite_team = fav
            fav_index = teams.index(fav)
            st.selectbox(
                "⭐ Favourite Team",
                options=teams,
                index=fav_index,
                key="fav_select",
                on_change=_change_favourite,
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
        elif page == "TodaysFixtures":
            from app.pages import todays_fixtures
            todays_fixtures.render(service)
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
