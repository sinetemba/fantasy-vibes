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

    def test_record_stores_blend_components(self):
        # p_* is the blended distribution shown to users; m_* is the raw
        # model and k_* the market — both kept so benchmarking stays honest.
        pred = _pred(0.45, 0.26, 0.29)
        pred["model_probabilities"] = {
            "home_win": 0.4, "draw": 0.25, "away_win": 0.35
        }
        pred["market_probabilities"] = {
            "home_win": 0.5, "draw": 0.27, "away_win": 0.23, "weight": 0.5
        }
        prediction_log.record("PL", [(self._fixture("9", "A", "B", 2), pred)])
        e = prediction_log._load()["PL|9"]
        self.assertAlmostEqual(e["p_away"], 0.29)
        self.assertEqual((e["m_home"], e["m_draw"], e["m_away"]), (0.4, 0.25, 0.35))
        self.assertEqual((e["k_home"], e["k_away"]), (0.5, 0.23))

    def test_record_without_blend_omits_components(self):
        prediction_log.record("PL", [
            (self._fixture("10", "A", "B", 2), _pred(0.5, 0.3, 0.2)),
        ])
        e = prediction_log._load()["PL|10"]
        self.assertNotIn("m_home", e)
        self.assertNotIn("k_home", e)

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

    def _settled(self, n, probs, result_for, league="PL", shown=None, raw=True):
        """Inject n settled log entries whose uncalibrated (r_*) probs are
        `probs` and whose displayed (p_*) probs are `shown` (default the
        same) — the calibration fit only reads settled rows, so they don't
        need record/settle. `raw=False` mimics entries logged before r_*."""
        shown = shown or probs
        entries = prediction_log._load()
        for i in range(n):
            e = {
                "league": league, "match_id": str(i),
                "home_team": "A", "away_team": "B",
                "date": (sast_now() - timedelta(days=1)).isoformat(),
                "predicted_at": sast_now().isoformat(), "model": "x",
                "p_home": shown[0], "p_draw": shown[1], "p_away": shown[2],
                "home_score": 1, "away_score": 0, "result": result_for(i),
            }
            if raw:
                e.update(r_home=probs[0], r_draw=probs[1], r_away=probs[2])
            entries[f"{league}|{i}"] = e

    def test_record_stores_raw_probabilities(self):
        pred = _pred(0.45, 0.28, 0.27)
        pred["raw_probabilities"] = {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}
        prediction_log.record("PL", [(self._fixture("11", "A", "B", 2), pred)])
        e = prediction_log._load()["PL|11"]
        self.assertEqual((e["r_home"], e["r_draw"], e["r_away"]), (0.6, 0.2, 0.2))
        self.assertAlmostEqual(e["p_home"], 0.45)

    def test_temperature_fits_raw_not_displayed(self):
        # Displayed probs were already flattened by an earlier T and look
        # calibrated; the raw model is still overconfident. Fitting on p_*
        # would return None and switch calibration off — the fit must keep
        # flagging the raw model's overconfidence.
        self._settled(
            150, (0.9, 0.05, 0.05),
            lambda i: "home_win" if i < 75 else ("draw" if i < 120 else "away_win"),
            shown=(0.5, 0.3, 0.2),
        )
        T = prediction_log.temperature("PL", min_n=100)
        self.assertIsNotNone(T)
        self.assertGreater(T, 1.0)

    def test_temperature_ignores_entries_without_raw(self):
        # Legacy entries carry only post-calibration p_*; they must not
        # count toward the sample floor or the fit.
        self._settled(
            150, (0.9, 0.05, 0.05),
            lambda i: "home_win" if i < 75 else "away_win",
            raw=False,
        )
        self.assertIsNone(prediction_log.temperature("PL", min_n=100))

    def test_temperature_none_below_sample_floor(self):
        self._settled(50, (0.5, 0.3, 0.2), lambda i: "home_win")
        self.assertIsNone(prediction_log.temperature("PL", min_n=120))

    def test_temperature_flags_overconfidence(self):
        # 50% hits at 90% confidence — the fit should flatten (T > 1).
        self._settled(
            150, (0.9, 0.05, 0.05),
            lambda i: "home_win" if i < 75 else "away_win",
        )
        T = prediction_log.temperature("PL", min_n=100)
        self.assertIsNotNone(T)
        self.assertGreater(T, 1.0)

    def test_temperature_near_one_when_calibrated(self):
        self._settled(
            150, (0.5, 0.3, 0.2),
            lambda i: "home_win" if i < 75 else ("draw" if i < 120 else "away_win"),
        )
        self.assertIsNone(prediction_log.temperature("PL", min_n=100))

    def test_temperature_pooled_fallback(self):
        # A league below the sample floor still calibrates when the pooled
        # log has enough settled entries — 150 overconfident BL1 rows give
        # a pooled T > 1 that thin-league "PL" inherits.
        self._settled(
            150, (0.9, 0.05, 0.05),
            lambda i: "home_win" if i < 75 else "away_win",
            league="BL1",
        )
        T = prediction_log.temperature("PL", min_n=100)
        self.assertIsNotNone(T)
        self.assertGreater(T, 1.0)

    def test_temperature_prefers_league_fit(self):
        # PL rows are well-calibrated (T ~ 1) while BL1 is overconfident;
        # with enough PL rows the pooled miscalibration must not leak in.
        self._settled(
            150, (0.5, 0.3, 0.2),
            lambda i: "home_win" if i < 75 else ("draw" if i < 120 else "away_win"),
        )
        self._settled(
            150, (0.9, 0.05, 0.05),
            lambda i: "home_win" if i < 75 else "away_win",
            league="BL1",
        )
        self.assertIsNone(prediction_log.temperature("PL", min_n=100))

    def test_record_stores_goals_markets_and_confidence(self):
        pred = _pred(0.6, 0.25, 0.15)
        pred.update({
            "over_under": {"1.5": {"over": 0.8}, "2.5": {"over": 0.55}, "3.5": {"over": 0.3}},
            "btts": {"yes": 0.5},
            "confidence": "high",
            "predicted_score": {"score": "2-0"},
        })
        prediction_log.record("PL", [(self._fixture("12", "A", "B", 2), pred)])
        e = prediction_log._load()["PL|12"]
        self.assertEqual((e["p_over15"], e["p_over25"], e["p_over35"]), (0.8, 0.55, 0.3))
        self.assertEqual(e["p_btts"], 0.5)
        self.assertEqual(e["confidence"], "high")
        self.assertEqual(e["predicted_score"], "2-0")

    def _settled_with(self, key, hs, aws, **fields):
        entries = prediction_log._load()
        entries[key] = {
            "league": "PL", "match_id": key, "home_team": "A", "away_team": "B",
            "date": (sast_now() - timedelta(days=1)).isoformat(),
            "predicted_at": sast_now().isoformat(), "model": "x",
            "p_home": 0.6, "p_draw": 0.25, "p_away": 0.15,
            "home_score": hs, "away_score": aws,
            "result": "home_win" if hs > aws else ("draw" if hs == aws else "away_win"),
            **fields,
        }

    def test_summary_scores_goals_and_confidence(self):
        # 3-1: over 2.5 yes, BTTS yes, home win (pick correct).
        self._settled_with("a", 3, 1, p_over25=0.7, p_btts=0.6, confidence="high")
        # 0-0: over 2.5 no, BTTS no, draw (pick wrong).
        self._settled_with("b", 0, 0, p_over25=0.6, p_btts=0.4, confidence="low")
        # Legacy entry: no goals fields - counts for outcomes only.
        self._settled_with("c", 1, 0)
        report = prediction_log.summary(days=30)
        self.assertEqual(report["n"], 3)
        over = report["goals"]["over_2.5"]
        self.assertEqual(over["n"], 2)
        self.assertAlmostEqual(over["hit_rate"], 0.5)
        self.assertAlmostEqual(over["base_rate"], 0.5)
        btts = report["goals"]["btts"]
        self.assertAlmostEqual(btts["hit_rate"], 1.0)
        self.assertEqual(report["by_confidence"]["high"], (1.0, 1))
        self.assertEqual(report["by_confidence"]["low"], (0.0, 1))
        lo, hi = report["hit_rate_ci"]
        self.assertTrue(0.0 <= lo < report["hit_rate"] < hi <= 1.0)

    def test_wilson_interval(self):
        lo, hi = prediction_log.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.404, places=2)
        self.assertAlmostEqual(hi, 0.596, places=2)
        self.assertEqual(prediction_log.wilson_interval(0, 0), (0.0, 1.0))

    def test_history_lists_settled_and_upcoming_newest_first(self):
        self._settled_with("old", 2, 0)
        prediction_log.record("PL", [(self._fixture("up", "C", "D", 3), _pred(0.3, 0.3, 0.4))])
        prediction_log.record("BL1", [(self._fixture("x", "E", "F", 3), _pred(0.5, 0.3, 0.2))])
        rows = prediction_log.history(days=30, league="PL")
        self.assertEqual([r["match_id"] for r in rows], ["up", "old"])
        self.assertIsNone(rows[0]["correct"])
        self.assertTrue(rows[1]["correct"])
        self.assertEqual(len(prediction_log.history(days=30)), 3)

    def test_unrated_teams_are_logged_and_counted(self):
        pred = dict(_pred(0.4, 0.3, 0.3), model="elo_fallback", unrated_teams=["Ghost FC"])
        prediction_log.record("INTF", [
            (self._fixture("u1", "Ghost FC", "B", 1), pred),
            (self._fixture("u2", "C", "Ghost FC", 2), pred),
            (self._fixture("u3", "C", "D", 2), _pred(0.4, 0.3, 0.3)),
        ])
        self.assertEqual(prediction_log.unrated_teams(days=30), [("INTF", "Ghost FC", 2)])

    def test_log_is_outside_the_cache_dir(self):
        self.assertNotEqual(prediction_log.LOG_PATH.parent.name, "cache")

    def test_legacy_log_is_migrated(self):
        legacy = prediction_log._legacy_path()
        legacy.parent.mkdir(parents=True)
        legacy.write_text('{"version": 1, "entries": {"PL|1": {"league": "PL"}}}', encoding="utf-8")
        self.assertIn("PL|1", prediction_log._load())
        self.assertFalse(legacy.exists())
        self.assertTrue(prediction_log.LOG_PATH.exists())

    def test_corrupt_log_is_set_aside_not_overwritten(self):
        prediction_log.LOG_PATH.write_text("{not json", encoding="utf-8")
        with self.assertLogs("src.data.prediction_log", "ERROR"):
            self.assertEqual(prediction_log._load(), {})
        aside = list(prediction_log.LOG_PATH.parent.glob("log.corrupt-*.json"))
        self.assertEqual(len(aside), 1)
        self.assertEqual(aside[0].read_text(encoding="utf-8"), "{not json")

    def test_daily_backup_taken_on_save(self):
        prediction_log.record("PL", [(self._fixture("1", "A", "B", 1), _pred(0.5, 0.3, 0.2))])
        prediction_log.record("PL", [(self._fixture("2", "C", "D", 1), _pred(0.5, 0.3, 0.2))])
        backups = list(prediction_log._backup_dir().glob("log-*.json"))
        self.assertEqual(len(backups), 1)  # one per day, and only once a file existed

    def test_retention_keeps_a_season(self):
        self.assertGreaterEqual(prediction_log.RETENTION_DAYS, 365)


if __name__ == "__main__":
    unittest.main()
