"""Tests for the prediction log (record, settle, summarise)."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

from src.data import prediction_log
from src.data.utils import sast_now


def _pred(home, draw, away):
    return {"model": "trained_poisson_elo",
            "outcome_probabilities": {"home_win": home, "draw": draw, "away_win": away}}


class TestPredictionLog(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patches = [
            mock.patch.object(prediction_log, "LOG_PATH", Path(tmp.name) / "log.json"),
            mock.patch.object(prediction_log, "_entries", None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _fixture(self, mid, home, away, days_ahead, **extra):
        when = sast_now() + timedelta(days=days_ahead)
        return {"match_id": mid, "home_team": home, "away_team": away,
                "date": when.replace(microsecond=0).isoformat(), "status": "scheduled",
                **extra}

    def _finish(self, m, hs, aws, days_ago=1):
        done = dict(m, status="full_time", home_score=hs, away_score=aws)
        done["date"] = (sast_now() - timedelta(days=days_ago)).replace(microsecond=0).isoformat()
        return done

    def test_only_future_fixtures_are_logged(self):
        prediction_log.record("PL", [
            (self._fixture("1", "Arsenal", "Chelsea", 2), _pred(0.5, 0.3, 0.2)),
            (self._fixture("2", "Leeds", "Spurs", -1), _pred(0.4, 0.3, 0.3)),
        ])
        self.assertEqual(list(prediction_log._load()), ["PL|1"])

    def test_settle_and_summary(self):
        a = self._fixture("1", "Arsenal", "Chelsea", 1)
        b = self._fixture("2", "Leeds", "Spurs", 1)
        prediction_log.record("PL", [(a, _pred(0.6, 0.25, 0.15)), (b, _pred(0.5, 0.3, 0.2))])
        # Results land a little later than the logged kickoff time.
        for e in prediction_log._load().values():
            e["date"] = (sast_now() - timedelta(days=1)).replace(microsecond=0).isoformat()
        prediction_log.settle("PL", [self._finish(a, 2, 0), self._finish(b, 0, 1)])

        report = prediction_log.summary(days=30)
        self.assertEqual(report["n"], 2)
        self.assertAlmostEqual(report["hit_rate"], 0.5)
        self.assertAlmostEqual(report["home_baseline"], 0.5)
        self.assertEqual(report["per_league"]["PL"], (0.5, 2))

    def test_same_fixture_from_another_feed_overwrites(self):
        prediction_log.record("PL", [
            (self._fixture("fd-1", "Manchester City", "Arsenal FC", 1), _pred(0.5, 0.3, 0.2)),
        ])
        prediction_log.record("PL", [
            (self._fixture("tsdb-9", "Man City", "Arsenal", 1), _pred(0.6, 0.2, 0.2)),
        ])
        entries = prediction_log._load()
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(next(iter(entries.values()))["p_home"], 0.6)

    def test_settles_across_feed_ids(self):
        m = self._fixture("fd-1", "Manchester City", "Arsenal FC", 1)
        prediction_log.record("PL", [(m, _pred(0.5, 0.3, 0.2))])
        played = self._finish(dict(m, match_id="tsdb-9", home_team="Man City",
                                   away_team="Arsenal"), 1, 1, days_ago=-1)
        prediction_log.settle("PL", [played])
        self.assertEqual(next(iter(prediction_log._load().values()))["result"], "draw")

    def test_persists_across_reload(self):
        prediction_log.record("PL", [(self._fixture("1", "A", "B", 1), _pred(0.4, 0.3, 0.3))])
        prediction_log._entries = None
        self.assertIn("PL|1", prediction_log._load())


if __name__ == "__main__":
    unittest.main()
