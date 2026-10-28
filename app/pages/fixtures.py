"""Fixtures page with filters and per-match predictions."""

from datetime import date, datetime, timedelta
from itertools import groupby
from typing import Any, Dict, List

MAX_BATCH_PREDICTIONS = 20
FIXTURES_PAGE_SIZE = 30

import streamlit as st

from app.ui import (
    apply_theme,
    display_disclaimer,
    display_header,
    form_badges_html,
    prediction_summary_html,
    show_warning_message,
)
from src.data import LeagueService
from src.data.utils import sast_now


def _parse_date(m: Dict[str, Any]) -> date:
    try:
        return datetime.fromisoformat(m["date"]).date()
    except Exception:
        return datetime.min.date()


def render(service: LeagueService):
    apply_theme()
    display_header(service.get_current_name(), "Full fixture list")

    st.markdown("## 📅 Fixtures")
    matches = service.get_matches()
    if not matches:
        show_warning_message("No fixtures available for this league.")
        return

    if "fixture_predictions" not in st.session_state:
        st.session_state["fixture_predictions"] = {}

    filtered = _render_filters(matches, service)

    if not filtered:
        st.info("No fixtures match the selected filters.")
        return

    # Page the list — each fixture is several widgets, so rendering a full
    # season at once makes every rerun slow. Any filter change starts over.
    signature = (
        service.get_current_code(),
        st.session_state.get("fixtures_round"),
        st.session_state.get("fixtures_use_date"),
        str(st.session_state.get("fixtures_date")),
        st.session_state.get("fixtures_this_week"),
        st.session_state.get("fixtures_show_all_upcoming"),
        st.session_state.get("fixtures_show_completed"),
    )
    if st.session_state.get("fixtures_page_sig") != signature:
        st.session_state["fixtures_page_sig"] = signature
        st.session_state["fixtures_limit"] = FIXTURES_PAGE_SIZE
    limit = st.session_state["fixtures_limit"]

    for date_label, day_matches in groupby(filtered[:limit], key=lambda m: _date_label(m)):
        st.markdown(f"### 📆 {date_label}")
        for m in day_matches:
            _render_fixture(service, m)
        st.markdown("---")

    if len(filtered) > limit:
        st.caption(f"Showing {limit} of {len(filtered)} fixtures")
        if st.button("⬇️ Show more", use_container_width=True, key="fixtures_show_more"):
            st.session_state["fixtures_limit"] = limit + FIXTURES_PAGE_SIZE
            st.rerun()

    display_disclaimer()


def _date_label(m: Dict[str, Any]) -> str:
    try:
        return datetime.fromisoformat(m["date"]).strftime("%a, %d %b %Y")
    except Exception:
        return "TBD"


def _render_filters(matches: List[Dict[str, Any]], service: LeagueService):
    current_code = service.get_current_code()
    if st.session_state.get("fixtures_league") != current_code:
        st.session_state["fixtures_league"] = current_code
        st.session_state.pop("fixtures_date", None)
        st.session_state.pop("fixtures_round", None)

    rounds = sorted({m.get("round", "") for m in matches if m.get("round")})
    dates = sorted({_parse_date(m) for m in matches if m.get("date")})

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        options = ["All rounds"] + rounds
        if "fixtures_round" in st.session_state and st.session_state["fixtures_round"] not in options:
            st.session_state["fixtures_round"] = "All rounds"
        st.selectbox("🏟️ Round", options, key="fixtures_round")
    with c2:
        st.checkbox("📅 Filter by date", value=False, key="fixtures_use_date")
        if dates and st.session_state.get("fixtures_use_date"):
            today = sast_now().date()
            default_date = today
            min_value = min(today, dates[0])
            max_value = max(today, dates[-1])
            if "fixtures_date" in st.session_state:
                stored = st.session_state["fixtures_date"]
                try:
                    stored_date = stored if isinstance(stored, date) else stored.date()
                    if not (min_value <= stored_date <= max_value):
                        st.session_state["fixtures_date"] = default_date
                except Exception:
                    st.session_state["fixtures_date"] = default_date
            st.date_input(
                "Date",
                value=default_date,
                min_value=min_value,
                max_value=max_value,
                key="fixtures_date",
                label_visibility="collapsed",
            )
    with c3:
        st.checkbox("🗓️ This week's fixtures", value=False, key="fixtures_this_week")
        st.checkbox("🔮 Show all upcoming", value=False, key="fixtures_show_all_upcoming")
        st.checkbox("🕓 Show completed", value=False, key="fixtures_show_completed")

    filtered = _apply_filters(matches)

    if filtered:
        if st.button(f"🔮 Predict first {MAX_BATCH_PREDICTIONS} scheduled", type="primary", use_container_width=True):
            predictions = st.session_state["fixture_predictions"]
            all_scheduled = [m for m in filtered if m.get("status") == "scheduled"]
            to_predict = all_scheduled[:MAX_BATCH_PREDICTIONS]
            with st.spinner(f"Predicting {len(to_predict)} fixtures..."):
                for m in to_predict:
                    pred = service.predict(m["home_team"], m["away_team"])
                    pred["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    predictions[m["match_id"]] = pred
            msg = f"Predicted {len(to_predict)} displayed matches"
            if len(all_scheduled) > MAX_BATCH_PREDICTIONS:
                msg += f" (capped at {MAX_BATCH_PREDICTIONS})"
            st.success(msg)

    if st.session_state["fixture_predictions"]:
        if st.button("🗑️ Clear predictions", use_container_width=True):
            st.session_state["fixture_predictions"] = {}
            st.rerun()

    return filtered


def _apply_filters(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    round_filter = st.session_state.get("fixtures_round", "All rounds")
    use_date = st.session_state.get("fixtures_use_date", False)
    date_filter = st.session_state.get("fixtures_date") if use_date else None
    show_completed = st.session_state.get("fixtures_show_completed", False)
    show_all_upcoming = st.session_state.get("fixtures_show_all_upcoming", False)
    this_week = st.session_state.get("fixtures_this_week", False)

    filtered = matches
    if round_filter != "All rounds":
        filtered = [m for m in filtered if m.get("round") == round_filter]

    if this_week:
        # Everything in the 7-day window starting today, regardless of status.
        start = sast_now().date()
        end = start + timedelta(days=6)
        filtered = [
            m for m in filtered
            if m.get("date") and start <= _parse_date(m) <= end
        ]
        filtered.sort(key=lambda m: m.get("date") or "")
        return filtered

    if date_filter:
        try:
            d = date_filter if isinstance(date_filter, date) else date_filter.date()
            filtered = [m for m in filtered if _parse_date(m) == d]
        except Exception:
            pass

    if show_all_upcoming:
        upcoming = [m for m in filtered if m.get("status") == "scheduled"]
        upcoming.sort(key=lambda m: m.get("date") or "")
        return upcoming

    upcoming = [m for m in filtered if m.get("status") == "scheduled"]
    completed = [m for m in filtered if m.get("status") == "full_time"]
    other = [m for m in filtered if m.get("status") not in ("scheduled", "full_time")]

    upcoming.sort(key=lambda m: m.get("date") or "")
    completed.sort(key=lambda m: m.get("date") or "", reverse=True)
    other.sort(key=lambda m: m.get("date") or "")

    result = upcoming + other
    if show_completed:
        result = completed + result
    return result


def _render_fixture(service: LeagueService, m: Dict[str, Any]):
    fixture_id = m["match_id"]
    predictions = st.session_state["fixture_predictions"]
    home = m["home_team"]
    away = m["away_team"]

    c_info, c_match, c_action = st.columns([1.4, 2.6, 1.2])

    with c_info:
        try:
            dt = datetime.fromisoformat(m["date"]) if m.get("date") else None
            time = dt.strftime("%H:%M") + " SAST" if dt else "—"
        except Exception:
            time = "—"
        if m.get("time_confirmed") is False:
            time = "TBC"
        st.markdown(
            f"""**{time}**  
<span style='color:#a0a0a0; font-size:0.8rem;'>{m.get('round', '')}</span>""",
            unsafe_allow_html=True,
        )

    with c_match:
        if m.get("status") == "full_time" and m.get("home_score") is not None:
            st.markdown(f"{home} {m['home_score']}-{m['away_score']} {away}")
        else:
            st.markdown(f"{home} vs {away}")
        st.caption(f"Source: {m.get('source', 'unknown')}")

    with c_action:
        if m.get("status") == "full_time":
            st.caption("Completed")
        elif st.button("🔮 Predict", key=f"predict_{fixture_id}", use_container_width=True):
            pred = service.predict(home, away)
            pred["predicted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            predictions[fixture_id] = pred
            st.rerun()

    if fixture_id in predictions:
        _render_prediction_summary(predictions[fixture_id], home, away, service=service)


def _render_prediction_summary(
    pred: Dict[str, Any], home: str, away: str, service: LeagueService = None
):
    st.markdown(prediction_summary_html(pred, home, away), unsafe_allow_html=True)
    if service is not None:
        st.markdown(
            f"<div style='font-size:0.8rem;color:#a0a0a0;margin-top:0.3rem;'>"
            f"🏠 Form: {form_badges_html(service.get_team_form(home))}"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"✈️ Form: {form_badges_html(service.get_team_form(away))}"
            f"&nbsp;<span style='font-size:0.7rem;'>(latest →)</span></div>",
            unsafe_allow_html=True,
        )
    if pred.get("predicted_at"):
        st.caption(f"Predicted on {pred['predicted_at']}")
