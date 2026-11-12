"""Tests for LeagueService helpers and composite source tracking."""

import unittest
from unittest import mock

from src.data import prediction_log
from src.data.league_service import (
    LeagueService,
    _memo_get,
    _memo_put,
    _results_signature,
)
from src.data.sources.composite import CompositeDataSource


class _Source:
    def __init__(self, name: str, matches=None, standings=None):
        self.name = name
        self._matches = matches or []
        self._standings = standings or []

    def is_available(self, league_config):
        return True

    def get_matches(self, league_config, season=None):
        return self._matches

    def get_standings(self, league_config, season=None):
        return self._standings


class TestCompositeSourceTracking(unittest.TestCase):
    """The composite records which source produced the data so callers
    don't have to re-query every provider to name it."""

    def test_matches_picks_first_available(self):
        composite = CompositeDataSource(
            match_sources=[_Source("source1"), _Source("source2", matches=[{"match_id": "1"}])],
            standings_sources=[],
        )
        matches = composite.get_matches({})
        self.assertEqual(matches, [{"match_id": "1"}])
        self.assertEqual(composite.last_matches_source, "source2")

    def test_source_name_none_when_all_empty(self):
        composite = CompositeDataSource(
            match_sources=[_Source("empty")],
            standings_sources=[_Source("empty")],
        )
        self.assertEqual(composite.get_matches({}), [])
        self.assertIsNone(composite.last_matches_source)
        self.assertEqual(composite.get_standings({}), [])
        self.assertIsNone(composite.last_standings_source)


def _ft(mid, hs=2, aws=1):
    return {
        "match_id": mid, "home_team": "A", "away_team": "B",
        "status": "full_time", "home_score": hs, "away_score": aws,
        "date": "2026-10-01T20:00:00",
    }


class TestResultsSignature(unittest.TestCase):
    """The engine-reuse shortcut: same results in → same signature out."""

    def test_unchanged_results_same_signature(self):
        matches = [_ft("x")]
        self.assertEqual(
            _results_signature(matches, [], {}), _results_signature(list(matches), [], {})
        )

    def test_new_result_changes_signature(self):
        self.assertNotEqual(
            _results_signature([_ft("x")], [], {}),
            _results_signature([_ft("x"), _ft("y", 0, 0)], [], {}),
        )

    def test_changed_score_changes_signature(self):
        self.assertNotEqual(
            _results_signature([_ft("x", 2, 1)], [], {}),
            _results_signature([_ft("x", 0, 0)], [], {}),
        )

    def test_related_pool_results_count(self):
        rel = {"CL": {"matches": [_ft("z")]}}
        self.assertNotEqual(
            _results_signature([_ft("x")], [], {}),
            _results_signature([_ft("x")], [], rel),
        )

    def test_scheduled_matches_ignored(self):
        sched = [{"match_id": "s", "status": "scheduled", "home_team": "A",
                  "away_team": "B", "date": "2026-10-02T20:00:00"}]
        self.assertEqual(
            _results_signature([_ft("x")], [], {}),
            _results_signature([_ft("x")] + sched, [], {}),
        )


class TestMatchMemo(unittest.TestCase):
    def test_roundtrip_returns_copies(self):
        _memo_put("ZZTEST", [{"match_id": "1", "home_team": "A"}])
        hit = _memo_get("ZZTEST")
        self.assertEqual(hit[0]["match_id"], "1")
        hit[0]["status"] = "mutated"
        self.assertNotIn("status", _memo_get("ZZTEST")[0])

    def test_empty_results_not_memoized(self):
        _memo_put("ZZEMPTY", [])
        self.assertIsNone(_memo_get("ZZEMPTY"))

    def test_unknown_code_misses(self):
        self.assertIsNone(_memo_get("ZZMISSING"))


class TestPredictFixtureLogging(unittest.TestCase):
    """Batch predict-alls defer their log write — one record() per league
    instead of one file rewrite per fixture."""

    def _service(self):
        svc = LeagueService.__new__(LeagueService)
        svc._current_code = "PL"
        svc.predict = lambda h, a, neutral=False: {"outcome_probabilities": {}}
        return svc

    def test_default_logs_prediction(self):
        svc = self._service()
        m = {"home_team": "A", "away_team": "B", "match_id": "m1"}
        with mock.patch.object(prediction_log, "record") as rec:
            svc.predict_fixture(m)
            rec.assert_called_once_with("PL", [(m, svc.predict("A", "B"))])

    def test_log_false_defers_write(self):
        svc = self._service()
        with mock.patch.object(prediction_log, "record") as rec:
            svc.predict_fixture({"home_team": "A", "away_team": "B", "match_id": "m1"}, log=False)
            rec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
