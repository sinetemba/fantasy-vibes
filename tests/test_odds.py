"""Tests for the football-data.co.uk odds benchmark."""

import math
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

from src.data import odds, prediction_log
from src.data.prediction_engine import teams_equivalent
from src.data.utils import sast_now


CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA,B365H,B365D,B365A
E0,{d0},Arsenal,Chelsea,2,1,H,1.80,3.60,4.50,1.75,3.50,4.40
E0,{d1},Man City,Liverpool,1,1,D,2.10,3.40,3.60,,,
E0,{d1},Nott'm Forest,West Brom,0,2,A,3.00,3.20,2.50,,,
"""


class TestOddsRows(unittest.TestCase):

    def _csv(self):
        d0 = (sast_now() - timedelta(days=2)).strftime("%d/%m/%Y")
        d1 = (sast_now() - timedelta(days=1)).strftime("%d/%m/%Y")
        return CSV.format(d0=d0, d1=d1)

    def test_parse_rows_prefers_market_average(self):
        with mock.patch.object(odds, "cached_get", return_value=self._csv()):
            rows = odds.odds_rows("PL", "2627")
        self.assertEqual(len(rows), 3)
        inv = [1 / 1.80, 1 / 3.60, 1 / 4.50]
        total = sum(inv)
        self.assertAlmostEqual(rows[0]["probs"][0], inv[0] / total)

    def test_falls_back_to_b365_when_average_missing(self):
        csv = self._csv().replace("2.10,3.40,3.60,,,", ",,,2.10,3.40,3.60")
        with mock.patch.object(odds, "cached_get", return_value=csv):
            rows = odds.odds_rows("PL", "2627")
        inv = [1 / 2.10, 1 / 3.40, 1 / 3.60]
        total = sum(inv)
        self.assertAlmostEqual(rows[1]["probs"][0], inv[0] / total)

    def test_unmapped_league_returns_empty(self):
        self.assertEqual(odds.odds_rows("PSL", "2627"), [])

    def test_season_code(self):
        from datetime import date
        self.assertEqual(odds._season_code(date(2026, 8, 1)), "2627")
        self.assertEqual(odds._season_code(date(2027, 2, 1)), "2627")

    def test_market_probs_lookup(self):
        with mock.patch.object(odds, "cached_get", return_value=self._csv()), \
             mock.patch.object(odds, "_rows_memo", {}):
            day = (sast_now() - timedelta(days=2)).isoformat()
            probs = odds.market_probs("PL", "Arsenal FC", "Chelsea", day)
        inv = [1 / 1.80, 1 / 3.60, 1 / 4.50]
        self.assertAlmostEqual(probs[0], inv[0] / sum(inv))

    def test_market_probs_none_for_uncovered(self):
        day = sast_now().isoformat()
        self.assertIsNone(odds.market_probs("PSL", "A", "B", day))
        self.assertIsNone(odds.market_probs("PL", "A", "B", None))


class TestFeedNames(unittest.TestCase):
    """football-data.co.uk spellings resolve to feed spellings."""

    def test_equivalences(self):
        pairs = [
            ("Nott'm Forest", "Nottingham Forest"),
            ("Ath Bilbao", "Athletic Club"),
            ("Ath Madrid", "Atlético Madrid"),
            ("Sp Gijon", "Sporting Gijón"),
            ("Espanol", "Espanyol"),
            ("Brest", "Stade Brestois"),
            ("Lyon", "Olympique Lyonnais"),
            ("Bayern Munich", "Bayern München"),
            ("Hamburg", "Hamburger SV"),
            ("West Brom", "West Bromwich Albion"),
            ("St Etienne", "Saint-Étienne"),
            ("Newcastle Utd", "Newcastle United"),
            ("Man City", "Manchester City"),
        ]
        for a, b in pairs:
            self.assertTrue(teams_equivalent(a, b), f"{a!r} != {b!r}")
            self.assertTrue(teams_equivalent(b, a), f"{b!r} != {a!r}")


class TestBenchmark(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patches = [
            mock.patch.object(prediction_log, "LOG_PATH", Path(tmp.name) / "log.json"),
            mock.patch.object(prediction_log, "_entries", None),
            # The parsed-CSV memo persists across tests — start each clean.
            mock.patch.object(odds, "_rows_memo", {}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _settled(self, mid, home, away, days_ago, probs, result):
        entries = prediction_log._load()
        entries[f"PL|{mid}"] = {
            "league": "PL", "match_id": mid,
            "home_team": home, "away_team": away,
            "date": (sast_now() - timedelta(days=days_ago)).isoformat(),
            "predicted_at": sast_now().isoformat(), "model": "x",
            "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2],
            "result": result,
        }

    def test_benchmark_scores_model_against_market(self):
        self._settled("1", "Arsenal FC", "Chelsea", 2, (0.6, 0.25, 0.15), "home_win")
        self._settled("2", "Manchester City", "Liverpool", 1, (0.5, 0.3, 0.2), "draw")
        self._settled("3", "Nottingham Forest", "West Bromwich Albion", 1,
                      (0.35, 0.3, 0.35), "away_win")
        csv = CSV.format(
            d0=(sast_now() - timedelta(days=2)).strftime("%d/%m/%Y"),
            d1=(sast_now() - timedelta(days=1)).strftime("%d/%m/%Y"),
        )
        with mock.patch.object(odds, "cached_get", return_value=csv):
            bm = odds.benchmark(days=30)
        self.assertEqual(bm["n"], 3)
        self.assertEqual(bm["total"], 3)
        # Model probabilities were chosen sharper than the market's on
        # every winner, so it should win the comparison here.
        self.assertLess(bm["model_logloss"], bm["market_logloss"])
        # Match 3: model's argmax ties home/away and misses; the market
        # prices away favourite and hits.
        self.assertAlmostEqual(bm["model_hit"], 1 / 3)
        self.assertAlmostEqual(bm["market_hit"], 2 / 3)

    def test_benchmark_splits_model_and_blend(self):
        # Entry logged post-blend: p_* is what users saw, m_* the raw model.
        entries = prediction_log._load()
        entries["PL|b"] = {
            "league": "PL", "match_id": "b",
            "home_team": "Arsenal", "away_team": "Chelsea",
            "date": (sast_now() - timedelta(days=2)).isoformat(),
            "predicted_at": sast_now().isoformat(), "model": "x",
            "p_home": 0.4, "p_draw": 0.23, "p_away": 0.37,
            "m_home": 0.6, "m_draw": 0.25, "m_away": 0.15,
            "k_home": 0.2, "k_draw": 0.2, "k_away": 0.6,
            "result": "home_win",
        }
        csv = CSV.format(
            d0=(sast_now() - timedelta(days=2)).strftime("%d/%m/%Y"),
            d1=(sast_now() - timedelta(days=1)).strftime("%d/%m/%Y"),
        )
        with mock.patch.object(odds, "cached_get", return_value=csv):
            bm = odds.benchmark(days=30)
        self.assertEqual(bm["n"], 1)
        self.assertEqual(bm["blend_n"], 1)
        # The headline model series uses the pre-blend probs, not p_*.
        self.assertAlmostEqual(bm["model_logloss"], -math.log(0.6))
        # The blend series scores the displayed (blended) probs on the
        # same fixture: blend 0.4 vs base 0.6 on a home_win result.
        self.assertAlmostEqual(bm["blend_logloss"], -math.log(0.4))
        self.assertAlmostEqual(bm["blend_base_logloss"], -math.log(0.6))

    def test_benchmark_empty_without_odds(self):
        self._settled("1", "Arsenal", "Chelsea", 1, (0.6, 0.25, 0.15), "home_win")
        with mock.patch.object(odds, "cached_get", return_value=None):
            bm = odds.benchmark(days=30)
        self.assertEqual(bm["n"], 0)
        self.assertEqual(bm["total"], 1)

    def test_skips_uncovered_leagues(self):
        entries = prediction_log._load()
        entries["PSL|9"] = {
            "league": "PSL", "match_id": "9",
            "home_team": "A", "away_team": "B",
            "date": (sast_now() - timedelta(days=1)).isoformat(),
            "predicted_at": sast_now().isoformat(), "model": "x",
            "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "result": "home_win",
        }
        bm = odds.benchmark(days=30)
        self.assertEqual(bm["n"], 0)
        self.assertEqual(bm["total"], 0)


if __name__ == "__main__":
    unittest.main()
