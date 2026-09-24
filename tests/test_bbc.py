"""Tests for the BBC source: event dedupe, TBC times and shared helpers."""

import unittest
from typing import Any, Dict, Optional

from src.data.league_service import LeagueService
from src.data.sources.bbc import BBCSource, _dedupe_events
from src.data.utils import compute_table, is_upcoming


def _event(
    event_id: str,
    start: str,
    certain: bool = True,
    rnd: Optional[str] = "Group A",
    home: str = "A",
    away: str = "B",
    tournament_id: str = "t1",
) -> Dict[str, Any]:
    return {
        "id": event_id,
        "startDateTime": start,
        "time": {"timeCertainty": certain},
        "status": "PreEvent",
        "home": {"id": home, "fullName": home},
        "away": {"id": away, "fullName": away},
        "tournament": {"id": tournament_id, "name": "T"},
        "round": {"id": rnd, "name": rnd} if rnd else None,
    }


class TestDedupeEvents(unittest.TestCase):
    """The feed emits a fixture as both a date-only placeholder and a
    timed event — sometimes with a date a day or two off."""

    def test_placeholders_collapse_into_timed_event(self):
        events = [
            (_event("p1", "2026-10-01", certain=False, rnd="GB"), "Group B"),
            (_event("p2", "2026-10-01", certain=False, rnd="GB"), "Group B"),
            (_event("t1", "2026-10-03T02:00:00Z", certain=True, rnd="GB"), "Group B"),
        ]
        out = _dedupe_events(events)
        self.assertEqual([e["id"] for e, _ in out], ["t1"])

    def test_placeholder_only_pair_keeps_one(self):
        events = [
            (_event("p1", "2026-09-24", certain=False), "G"),
            (_event("p2", "2026-09-24", certain=False), "G"),
        ]
        out = _dedupe_events(events)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0]["id"], "p1")

    def test_same_pairing_within_a_day_merges_across_rounds(self):
        events = [
            (_event("a", "2026-10-01T22:00:00Z", rnd="G1"), "G1"),
            (_event("b", "2026-10-02T02:00:00Z", rnd="G2"), "G2"),
        ]
        self.assertEqual(len(_dedupe_events(events)), 1)

    def test_two_timed_fixtures_in_same_round_stay(self):
        events = [
            (_event("a", "2026-10-01T02:00:00Z"), "G"),
            (_event("b", "2026-10-20T02:00:00Z"), "G"),
        ]
        self.assertEqual(len(_dedupe_events(events)), 2)

    def test_swapped_home_away_not_merged(self):
        events = [
            (_event("a", "2026-10-01"), "G"),
            (_event("b", "2026-10-01", home="B", away="A"), "G"),
        ]
        self.assertEqual(len(_dedupe_events(events)), 2)

    def test_different_tournaments_not_merged(self):
        events = [
            (_event("a", "2026-10-01", tournament_id="t1"), "G"),
            (_event("b", "2026-10-01", tournament_id="t2"), "G"),
        ]
        self.assertEqual(len(_dedupe_events(events)), 2)


class TestToMatch(unittest.TestCase):
    def test_unconfirmed_time_renders_tbc(self):
        src = BBCSource()
        m = src._to_match(_event("x", "2026-10-01", certain=False), "G")
        self.assertEqual(m["minutes_elapsed"], "TBC")
        self.assertFalse(m["time_confirmed"])

    def test_confirmed_time_renders_sast(self):
        src = BBCSource()
        m = src._to_match(_event("x", "2026-10-01T20:00:00Z", certain=True), "G")
        self.assertEqual(m["minutes_elapsed"], "22:00 SAST")
        self.assertTrue(m["time_confirmed"])

    def test_full_time_parses_score(self):
        e = _event("x", "2026-10-01T20:00:00Z")
        e["status"] = "PostEvent"
        e["home"]["score"] = "2"
        e["away"]["score"] = "1"
        m = BBCSource()._to_match(e, "G")
        self.assertEqual(m["status"], "full_time")
        self.assertEqual((m["home_score"], m["away_score"]), (2, 1))


class _FakeSource:
    name = "fake"

    def __init__(self, by_code):
        self._by_code = by_code

    def get_matches(self, cfg, season=None):
        return list(self._by_code.get(cfg["name"], []))


def _match(home, away, day, comp=""):
    return {
        "match_id": f"{home}_{away}_{day}",
        "home_team": home,
        "away_team": away,
        "date": f"2026-10-{day}T20:00:00",
        "status": "scheduled",
        "source": "fake",
    }


class TestGroupMatchDedupe(unittest.TestCase):
    def _service(self, leagues, by_code):
        svc = LeagueService.__new__(LeagueService)
        svc._leagues = leagues
        svc._source = _FakeSource(by_code)
        svc._group_match_cache = {}
        return svc

    def test_cross_competition_dupes_collapse(self):
        leagues = {
            "CNL": {"name": "Nations League", "group": "international"},
            "WCQ": {"name": "Qualifiers", "group": "international"},
        }
        by_code = {
            "Nations League": [_match("Mexico", "Jamaica", "05")],
            "Qualifiers": [_match("Mexico", "Jamaica", "05")],
        }
        svc = self._service(leagues, by_code)
        out = svc.get_group_matches("international")
        self.assertEqual(len(out), 1)

    def test_intf_entry_loses_to_specific_competition(self):
        leagues = {
            "INTF": {"name": "Friendlies", "group": "international"},
            "CNL": {"name": "Nations League", "group": "international"},
        }
        by_code = {
            "Friendlies": [_match("Mexico", "Jamaica", "05")],
            "Nations League": [_match("Mexico", "Jamaica", "05")],
        }
        svc = self._service(leagues, by_code)
        out = svc.get_group_matches("international")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["competition"], "CNL")

    def test_distinct_fixtures_untouched(self):
        leagues = {
            "CNL": {"name": "Nations League", "group": "international"},
            "WCQ": {"name": "Qualifiers", "group": "international"},
        }
        by_code = {
            "Nations League": [_match("Mexico", "Jamaica", "05")],
            "Qualifiers": [_match("Canada", "Haiti", "05")],
        }
        svc = self._service(leagues, by_code)
        self.assertEqual(len(svc.get_group_matches("international")), 2)


class TestHelpers(unittest.TestCase):
    def test_is_upcoming(self):
        self.assertFalse(is_upcoming({"status": "scheduled", "date": "2020-01-01T15:00:00"}))
        self.assertTrue(is_upcoming({"status": "scheduled", "date": "2999-01-01T15:00:00"}))
        self.assertFalse(is_upcoming({"status": "scheduled", "date": None}))
        self.assertFalse(is_upcoming({"status": "full_time", "date": "2999-01-01"}))

    def test_compute_table(self):
        matches = [
            {"home_team": "A", "away_team": "B", "status": "full_time", "home_score": 2, "away_score": 0},
            {"home_team": "B", "away_team": "C", "status": "full_time", "home_score": 1, "away_score": 1},
            {"home_team": "A", "away_team": "C", "status": "scheduled"},
        ]
        table = compute_table(matches)
        # A won (3 pts); B and C drew each other (1 pt) but C has the better
        # goal difference. The scheduled A–C game doesn't count.
        self.assertEqual([r["team"] for r in table], ["A", "C", "B"])
        self.assertEqual(table[0]["points"], 3)
        self.assertEqual(table[1]["played"], 1)
        self.assertEqual(table[0]["form"], "W")


if __name__ == "__main__":
    unittest.main()
