"""Main entry point for the Fantasy Vibes multi-league football app."""

import json
import logging
import sys
import time
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

from app.ui import (
    apply_theme,
    get_league_service,
    is_current_service,
    prewarm_all_services,
)
from src.data import LeagueService
from src.data.league_service import LEAGUES_PATH

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

INTERNATIONAL_CODE = "INTL"

PAGES = {
    "🏠 Home": "Home",
    "⚽ Live": "Live",
    "📅 Fixtures": "Fixtures",
    "🗓️ Today's Fixtures": "TodaysFixtures",
    "🌍 International": "International",
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
                    "favourite_teams": st.session_state.get("favourite_teams", {}),
                },
                f,
                indent=2,
            )
    except Exception:
        pass


def init_session_state():
    svc = st.session_state.get("league_service")
    if svc is None:
        st.session_state.league_service = get_league_service(_default_league_code())
    elif not is_current_service(svc):
        # Session holds a service from before a code reload — swap in a
        # current one for the same league.
        st.session_state.league_service = get_league_service(svc.get_current_code())
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "last_league" not in st.session_state:
        st.session_state.last_league = st.session_state.league_service.get_current_code()
    if "favourite_teams" not in st.session_state:
        # One favourite per league code. Older preference files hold a
        # single "favourite_team", which belongs to their last league.
        prefs = _load_preferences()
        favs = dict(prefs.get("favourite_teams") or {})
        if prefs.get("favourite_team") and prefs.get("last_league"):
            favs.setdefault(prefs["last_league"], prefs["favourite_team"])
        service = st.session_state.league_service
        fav = st.query_params.get("fav")
        if fav and fav in service.get_teams():
            favs[service.get_current_code()] = fav
        st.session_state.favourite_teams = favs
    # Starts the shared warm-up/refresh thread on first call in the process;
    # every later call (any session) is a no-op.
    prewarm_all_services()


def _default_league_code() -> str:
    prefs = _load_preferences()
    candidate = st.query_params.get("league") or prefs.get("last_league")
    try:
        # Read the registry directly — building a service just to pick a
        # league code would trigger a needless full data fetch.
        with open(LEAGUES_PATH, "r", encoding="utf-8") as f:
            leagues = json.load(f)
        selectable = [
            (code, cfg) for code, cfg in leagues.items() if not cfg.get("group")
        ]
        if candidate and candidate in [c for c, _ in selectable]:
            return candidate
        for code, cfg in selectable:
            if cfg.get("default"):
                return code
        return selectable[0][0]
    except Exception:
        return candidate or "PL"


def _change_league():
    new_code = st.session_state.get("league_select")
    if not new_code:
        return
    # "International" is a pseudo-entry that opens the international page.
    if new_code == INTERNATIONAL_CODE:
        st.session_state.current_page = "International"
        return
    if st.session_state.current_page == "International":
        st.session_state.current_page = "Home"
    if new_code != st.session_state.league_service.get_current_code():
        # Swap to the shared cached service rather than mutating it.
        st.session_state.league_service = get_league_service(new_code)
        st.session_state.last_league = new_code
        st.query_params["league"] = new_code
        _save_preferences()
        # Predictions are league-specific; stale ones cause display errors.
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("quick_prediction", None)
        st.session_state.pop("fixture_predictions", None)
        st.session_state.pop("intl_predictions", None)


def _change_favourite(code: str):
    team = st.session_state.get(f"fav_select_{code}")
    favs = st.session_state.favourite_teams
    if team:
        favs[code] = team
    else:
        favs.pop(code, None)
    _save_preferences()


def _updated_ago(ts) -> str:
    if not ts:
        return "not loaded yet"
    secs = max(0, time.time() - ts)
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    return f"{int(secs // 3600)} h ago"


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

        # League selector — "International" is a pseudo-entry that opens the
        # international page rather than switching the league context.
        leagues = service.get_leagues()
        names = dict(leagues)
        names[INTERNATIONAL_CODE] = "🌍 International"
        options = [c for c, _ in leagues] + [INTERNATIONAL_CODE]
        on_intl = st.session_state.current_page == "International"
        st.session_state["league_select"] = (
            INTERNATIONAL_CODE if on_intl else service.get_current_code()
        )
        st.selectbox(
            "🏆 Select League",
            options=options,
            format_func=lambda x: names.get(x, x),
            key="league_select",
            on_change=_change_league,
        )

        # Favourite team for the current league — Home shows its form, next
        # match and a one-click prediction.
        if not on_intl:
            code = service.get_current_code()
            teams = service.get_teams()
            current = st.session_state.favourite_teams.get(code)
            fav_options = [None] + teams
            st.selectbox(
                "⭐ Favourite team",
                options=fav_options,
                index=fav_options.index(current) if current in teams else 0,
                format_func=lambda t: "— none —" if t is None else t,
                key=f"fav_select_{code}",
                on_change=_change_favourite,
                args=(code,),
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

        # Freshness, not provider names — those are listed on the About page.
        st.caption(f"🕒 Data updated {_updated_ago(getattr(service, 'last_refreshed', None))}")

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
        elif page == "International":
            from app.pages import international
            international.render(service)
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
