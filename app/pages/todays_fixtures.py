"""Today's fixtures page showing matches across all configured leagues."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from app.ui import (
    COLORS,
    apply_theme,
    display_disclaimer,
    display_header,
    get_league_service,
    match_card_html,
    probability_bar,
)
from src.data import LeagueService
from src.data.league_service import match_importance
from src.data.prediction_engine import _norm_key
from src.data.sources.bbc import BBCSource
from src.data.sources.composite import CompositeDataSource
from src.data.sources.fixture_download import FixtureDownloadSource
from src.data.sources.football_data import FootballDataSource
from src.data.sources.openfootball import OpenfootballSource
from src.data.sources.psl_scraper import PSLScraperSource
from src.data.sources.thesportsdb import TheSportsDBSource
from src.data.utils import bust_live_cache, sast_now

logger = logging.getLogger(__name__)


def _match_date(m: Dict[str, Any]) -> date:
    try:
        return datetime.fromisoformat(m["date"]).date()
    except Exception:
        return date.min


@st.cache_data(ttl=3600, show_spinner=False)
def _load_thesportsdb_id_to_code() -> Dict[str, str]:
    """Map TheSportsDB league ids to configured league codes."""
    path = Path(__file__).resolve().parent.parent.parent / "data" / "leagues.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            leagues = json.load(f)
    except Exception:
        return {}
    return {
        str(cfg["thesportsdb"]["league_id"]): code
        for code, cfg in leagues.items()
        if "thesportsdb" in cfg and cfg["thesportsdb"].get("league_id")
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _load_leagues() -> Dict[str, Dict[str, Any]]:
    """Load the leagues registry."""
    path = Path(__file__).resolve().parent.parent.parent / "data" / "leagues.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _load_todays_fixtures(today: str) -> List[Dict[str, Any]]:
    target = date.fromisoformat(today)
    fixtures: List[Dict[str, Any]] = []
    leagues = _load_leagues()

    # TheSportsDB eventsday gives all soccer fixtures for this date (free demo key works).
    try:
        tsdb_matches = TheSportsDBSource().get_matches_by_date(target)
        if tsdb_matches:
            id_to_code = _load_thesportsdb_id_to_code()
            for m in tsdb_matches:
                tsdb_id = str(m.get("league_code", ""))
                if tsdb_id not in id_to_code:
                    continue  # only configured leagues — eventsday covers everything
                m["league_code"] = id_to_code[tsdb_id]
                m["league_name"] = leagues.get(m["league_code"], {}).get("name", "")
                fixtures.append(m)
    except Exception as exc:
        logger.warning(f"TheSportsDB eventsday failed: {exc}")

    # Football-Data.org for configured leagues (often has scheduled fixtures TheSportsDB misses).
    try:
        fd_source = FootballDataSource()
        entries = [
            (code, cfg) for code, cfg in leagues.items() if "football_data" in cfg
        ]

        def _fd_rows(item):
            code, cfg = item
            try:
                fd_matches = fd_source.get_matches(
                    cfg,
                    date_from=today,
                    date_to=today,
                    ttl_seconds=1800,
                )
            except Exception as exc:
                logger.warning(f"Football-Data fixtures for {code} failed: {exc}")
                return []
            rows = []
            for m in fd_matches:
                row = dict(m)
                row["league_code"] = code
                row["league_name"] = cfg["name"]
                rows.append(row)
            return rows

        with ThreadPoolExecutor(max_workers=min(6, len(entries) or 1)) as pool:
            for rows in pool.map(_fd_rows, entries):
                fixtures.extend(rows)
    except Exception as exc:
        logger.warning(f"Football-Data fixtures failed: {exc}")

    # Pull every configured league through the source cascade so the whole
    # dropdown is covered — PSL (scraper) and all international competitions
    # (BBC) that eventsday/Football-Data miss.
    source = CompositeDataSource(
        match_sources=[
            FixtureDownloadSource(),
            BBCSource(),
            FootballDataSource(),
            OpenfootballSource(),
            PSLScraperSource(),
            TheSportsDBSource(),
        ],
        standings_sources=[],
    )

    def _league_rows(item):
        code, cfg = item
        try:
            league_matches = source.get_matches(cfg)
        except Exception as exc:
            logger.warning(f"{code} fixtures failed: {exc}")
            return []
        rows = []
        for m in league_matches:
            if _match_date(m) != target:
                continue
            row = dict(m)
            row["league_code"] = code
            row["league_name"] = cfg.get("name", code)
            rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=6) as pool:
        for rows in pool.map(_league_rows, leagues.items()):
            fixtures.extend(rows)

    if fixtures:
        # Dedupe by a stable key (best-effort across sources). Sources can
        # disagree on kickoff time by minutes and on team-name decoration
        # ("Manchester United FC" vs "Manchester United"), so normalise
        # names and compare on the date part only.
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for m in fixtures:
            key = (
                _norm_key(m.get("home_team", "")),
                _norm_key(m.get("away_team", "")),
                (m.get("date") or "")[:10],
            )
            if key not in seen:
                seen.add(key)
                unique.append(m)
        unique.sort(key=lambda m: (m.get("league_name", ""), m.get("date") or ""))
        return unique

    # Fallback: iterate configured leagues and filter by today's date.
    fallback: List[Dict[str, Any]] = []
    base = get_league_service("PL")
    for code, name in base.get_leagues():
        try:
            league_service = get_league_service(code)
        except Exception as exc:
            logger.error(f"Could not load league {code}: {exc}")
            continue
        for m in league_service.get_matches():
            if _match_date(m) != target:
                continue
            row = dict(m)
            row["league_code"] = code
            row["league_name"] = name
            fallback.append(row)
    fallback.sort(key=lambda m: (m["league_name"], m.get("date") or ""))
    return fallback


def _league_service_for(code: str) -> LeagueService:
    """Return a cached LeagueService for a configured league code."""
    return get_league_service(code)


def _predict_all_matches(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate predictions for every match. Falls back to a generic model for unconfigured leagues."""
    predictions: Dict[str, Any] = {}
    fallback_svc = None
    for m in matches:
        code = m.get("league_code")
        if not code:
            continue
        try:
            svc = _league_service_for(code)
            pred = svc.predict(m["home_team"], m["away_team"])
            predictions[m["match_id"]] = pred
        except ValueError as exc:
            if "Unknown league code" not in str(exc):
                logger.warning(f"Prediction failed for {m.get('match_id')}: {exc}")
                continue
            # Unconfigured TheSportsDB league: use a fallback model so every fixture gets a prediction.
            if fallback_svc is None:
                try:
                    fallback_svc = _league_service_for("PL")
                except Exception as inner:
                    logger.warning(f"Could not create fallback prediction service: {inner}")
                    continue
            pred = fallback_svc.predict(m["home_team"], m["away_team"])
            predictions[m["match_id"]] = pred
        except Exception as exc:
            logger.warning(f"Prediction failed for {m.get('match_id')}: {exc}")
    return predictions


def _clear_cache():
    """Force today's fixtures to refetch: clear the page memoisation and open
    a live-refresh window so source calls bypass their TTLs. Cache files stay
    intact — other pages keep their cached data."""
    _load_todays_fixtures.clear()
    bust_live_cache(30)


def _top_games(matches: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """Marquee fixtures for the day, ranked by shared importance scoring
    (national-team ratings + competition bonus). Only live/scheduled games
    count, and zero-scoring fixtures are dropped so the section hides on
    plain domestic slates."""
    watchable = [m for m in matches if m.get("status") in ("live", "half_time", "scheduled")]
    ranked = sorted(watchable, key=match_importance, reverse=True)
    return [m for m in ranked if match_importance(m) > 0][:n]


def render(service: LeagueService):
    apply_theme()

    today = sast_now().date()
    with st.spinner("Loading today's fixtures..."):
        matches = _load_todays_fixtures(today.isoformat())

    today_str = today.strftime("%A, %d %B %Y")
    if matches:
        leagues = sorted({m["league_name"] for m in matches})
        subtitle = f"{', '.join(leagues)} - {today_str}"
    else:
        subtitle = f"All leagues - {today_str}"
    display_header("Today's Fixtures", subtitle)

    if not matches:
        return

    if "today_predictions" not in st.session_state:
        st.session_state["today_predictions"] = {}

    # Metrics reflect all loaded fixtures.
    live_count = sum(1 for m in matches if m.get("status") == "live")
    done_count = sum(1 for m in matches if m.get("status") == "full_time")
    scheduled_count = sum(1 for m in matches if m.get("status") == "scheduled")

    cols = st.columns(3)
    cols[0].metric("Live", live_count)
    cols[1].metric("Completed", done_count)
    cols[2].metric("Scheduled", scheduled_count)

    filter_cols = st.columns(3)
    with filter_cols[0]:
        show_played = st.toggle("Played", value=True, key="today_show_played")
    with filter_cols[1]:
        show_scheduled = st.toggle("Scheduled", value=True, key="today_show_scheduled")
    with filter_cols[2]:
        show_live = st.toggle("Live", value=True, key="today_show_live")

    status_groups = {
        "played": ["full_time"],
        "live": ["live", "half_time"],
        "scheduled": ["scheduled", "postponed", "cancelled"],
    }
    visible_matches = [
        m
        for m in matches
        if (
            (m.get("status") in status_groups["played"] and show_played)
            or (m.get("status") in status_groups["live"] and show_live)
            or (m.get("status") in status_groups["scheduled"] and show_scheduled)
        )
    ]

    if not visible_matches:
        st.info("No fixtures match the selected filters.")
        return

    c1, c2, c3 = st.columns([4, 1, 1])
    with c1:
        st.markdown("Fixture list aggregated across all configured leagues for the current day.")
    with c2:
        if st.button("🔄 Refresh", use_container_width=True):
            _clear_cache()
            st.session_state["today_predictions"] = {}
            st.rerun()
    with c3:
        if st.button("🔮 Regenerate Predictions", use_container_width=True):
            with st.spinner("Predicting selected fixtures..."):
                st.session_state["today_predictions"].update(_predict_all_matches(visible_matches))
            st.rerun()

    # Auto-generate predictions for the currently selected fixtures on first load.
    if not st.session_state["today_predictions"]:
        with st.spinner("Generating predictions for selected fixtures..."):
            st.session_state["today_predictions"] = _predict_all_matches(visible_matches)

    predictions = st.session_state.get("today_predictions", {})

    # Marquee fixtures pinned above the full list — on international weeks
    # this surfaces the strongest national-team matchups; on club nights the
    # CL/EL bonus keeps European games at the top.
    top = _top_games(matches)
    if top:
        st.markdown("### ⭐ Top Games")
        for m in top:
            _render_match(m, predictions)
        st.markdown("---")

    if show_live and not show_played and not show_scheduled:
        # Live-only view: flat list ranked by importance — same scoring as
        # the Top Games section (national ratings + competition bonus).
        ranked_live = sorted(visible_matches, key=match_importance, reverse=True)
        st.markdown(f"### 🔴 Live Now ({len(ranked_live)})")
        for m in ranked_live:
            _render_match(m, predictions)
    else:
        for league_name, league_matches in groupby(visible_matches, key=lambda m: m["league_name"]):
            league_list = list(league_matches)
            st.markdown(f"### 🏆 {league_name} ({len(league_list)})")
            for m in league_list:
                _render_match(m, predictions)

    display_disclaimer()


def _render_match(m: Dict[str, Any], predictions: Optional[Dict[str, Any]] = None):
    time = ""
    if m.get("date"):
        try:
            dt = datetime.fromisoformat(m["date"])
            time = dt.strftime("%H:%M")
        except Exception:
            pass

    minutes = m.get("minutes_elapsed", "TBD")
    if m.get("status") == "full_time":
        minutes = "FT"
    elif m.get("status") == "live":
        minutes = minutes or "LIVE"

    st.markdown(
        match_card_html(
            m["home_team"],
            m["away_team"],
            m.get("home_score"),
            m.get("away_score"),
            m.get("status", "scheduled"),
            str(minutes),
            round_label=m.get("round", ""),
            time=time,
        ),
        unsafe_allow_html=True,
    )

    pred = predictions.get(m.get("match_id")) if predictions else None
    if pred:
        outcome = pred.get("outcome_probabilities", {})
        score = pred.get("predicted_score", {})
        expected = pred.get("expected_goals", {})
        st.caption(
            f"Predicted score: **{score.get('score', 'N/A')}**  |  "
            f"xG: {expected.get('home', 0):.2f} - {expected.get('away', 0):.2f}"
        )
        home_win = outcome.get("home_win", 0)
        away_win = outcome.get("away_win", 0)
        draw = outcome.get("draw", 0)
        # Favourite (higher win probability) gets the green bar; underdog the red.
        if home_win >= away_win:
            home_color, away_color = COLORS["home_win"], COLORS["away_win"]
        else:
            home_color, away_color = COLORS["away_win"], COLORS["home_win"]
        c1, c2, c3 = st.columns(3)
        with c1:
            probability_bar(f"🏠 {m['home_team']}", home_win, home_color)
        with c2:
            probability_bar("🤝 Draw", draw, COLORS["draw"])
        with c3:
            probability_bar(f"✈️ {m['away_team']}", away_win, away_color)
