"""Tests for the pure prediction-display helpers in app.ui and the
Track Record history table."""

import unittest

from app.pages.track_record import history_rows
from app.ui import (
    confidence_badge_html,
    confidence_level,
    goals_text,
    over_under_rows,
    prediction_caveat,
    prediction_drivers,
    prediction_verdict,
)


def _probs(h, d, a):
    return {"home_team": "Arsenal", "away_team": "Chelsea",
            "outcome_probabilities": {"home_win": h, "draw": d, "away_win": a}}


class TestVerdict(unittest.TestCase):

    def test_draw_in_play_is_a_close_call(self):
        self.assertEqual(prediction_verdict(_probs(0.40, 0.29, 0.31)), "Close call: Arsenal or a draw")

    def test_even_sides_with_unlikely_draw_is_coin_flip(self):
        self.assertEqual(prediction_verdict(_probs(0.43, 0.15, 0.42)), "Coin flip")

    def test_clear_favourite_unchanged(self):
        self.assertEqual(prediction_verdict(_probs(0.70, 0.18, 0.12)), "Arsenal strong favourites")

    def test_draw_leading(self):
        self.assertEqual(prediction_verdict(_probs(0.30, 0.40, 0.30)), "Draw the likeliest result")


class TestUnratedCaveat(unittest.TestCase):

    def test_caveat_names_unrated_team(self):
        pred = dict(_probs(0.5, 0.3, 0.2), model="elo_fallback", unrated_teams=["Ghost FC"])
        self.assertIn("no rating for Ghost FC", prediction_caveat(pred))


def _pred(**extra):
    pred = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "model": "trained_poisson_elo",
        "outcome_probabilities": {"home_win": 0.6, "draw": 0.25, "away_win": 0.15},
        "over_under": {
            "1.5": {"over": 0.8, "under": 0.2},
            "3.5": {"over": 0.3, "under": 0.7},
            "2.5": {"over": 0.55, "under": 0.45},
        },
        "btts": {"yes": 0.51, "no": 0.49},
        "confidence": "high",
        "drivers": {
            "elo": {"home": 1650, "away": 1580, "home_field": 70},
            "form_context": {"home": 1.08, "away": 0.96},
            "components": {
                "goal_model": {"home_win": 0.58, "draw": 0.24, "away_win": 0.18},
                "rating_model": {"home_win": 0.62, "draw": 0.26, "away_win": 0.12},
                "goal_model_weight": 0.5,
            },
        },
    }
    pred.update(extra)
    return pred


class TestPredictionDisplay(unittest.TestCase):

    def test_goals_text(self):
        self.assertEqual(goals_text(_pred()), "Over 2.5 goals 55% · Both teams score 51%")

    def test_goals_text_legacy_prediction(self):
        pred = _pred(over_under_2_5={"over": 0.4, "under": 0.6})
        del pred["over_under"]
        self.assertTrue(goals_text(pred).startswith("Over 2.5 goals 40%"))

    def test_over_under_rows_sorted_by_line(self):
        self.assertEqual([r[0] for r in over_under_rows(_pred())], ["1.5", "2.5", "3.5"])

    def test_confidence_badge(self):
        self.assertIn("High confidence", confidence_badge_html(_pred()))
        self.assertEqual(confidence_badge_html(_pred(confidence=None)), "")

    def test_caveat_downgrades_confidence(self):
        pred = _pred(model="elo_fallback")
        self.assertTrue(prediction_caveat(pred))
        self.assertEqual(confidence_level(pred), "low")

    def test_drivers(self):
        text = " ".join(prediction_drivers(_pred()))
        self.assertIn("Arsenal 1650 vs Chelsea 1580", text)
        self.assertIn("70-point gap for Arsenal", text)
        self.assertIn("Arsenal +8%", text)
        self.assertIn("Chelsea -4%", text)
        self.assertIn("goals model 58/24/18", text)

    def test_drivers_mention_calibration_and_market(self):
        pred = _pred(market_probabilities={
            "home_win": 0.5, "draw": 0.27, "away_win": 0.23, "weight": 0.5
        })
        pred["drivers"]["calibration_T"] = 1.2
        text = " ".join(prediction_drivers(pred))
        self.assertIn("overconfident", text)
        self.assertIn("Bookmaker odds", text)

    def test_fallback_drivers(self):
        pred = _pred(model="elo_fallback")
        pred["drivers"] = {"elo": {"home": 1500, "away": 1400, "home_field": 0},
                           "form_context": None, "components": None}
        text = " ".join(prediction_drivers(pred))
        self.assertIn("neutral venue", text)
        self.assertIn("Ratings only", text)


class TestHistoryRows(unittest.TestCase):

    def test_settled_and_pending_rows(self):
        base = {"league": "PL", "home_team": "A", "away_team": "B",
                "date": "2026-11-01T17:30:00", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
        settled = dict(base, result="away_win", home_score=0, away_score=1,
                       correct=False, confidence="medium", p_over25=0.45)
        pending = dict(base, result=None, correct=None)
        rows = history_rows([settled, pending])
        self.assertEqual(rows[0]["Result"], "0-1")
        self.assertEqual(rows[0]["Correct"], "❌")
        self.assertEqual(rows[0]["Pick"], "Home")
        self.assertEqual(rows[0]["Confidence"], "Medium")
        self.assertEqual(rows[0]["Over 2.5"], "45%")
        self.assertEqual(rows[0]["Kickoff"], "2026-11-01 17:30")
        self.assertEqual(rows[1]["Correct"], "⏳")
        self.assertEqual(rows[1]["BTTS"], "—")


if __name__ == "__main__":
    unittest.main()
