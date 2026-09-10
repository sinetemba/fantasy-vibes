"""Basic tests for the prediction engine."""

import unittest

from src.data.prediction_engine import PredictionEngine


class TestPredictionEngine(unittest.TestCase):

    def test_not_fitted_without_data(self):
        engine = PredictionEngine()
        engine.fit([])
        self.assertFalse(engine.fitted)
        self.assertIsNone(engine.last_trained)

    def test_fits_and_predicts(self):
        matches = [
            {
                "status": "full_time",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 2,
                "away_score": 1,
                "date": "2026-08-01T15:00:00",
            },
            {
                "status": "full_time",
                "home_team": "Team B",
                "away_team": "Team A",
                "home_score": 1,
                "away_score": 1,
                "date": "2026-08-08T15:00:00",
            },
        ]
        engine = PredictionEngine()
        engine.fit(matches)
        self.assertTrue(engine.fitted)
        self.assertEqual(len(engine.teams), 2)

        pred = engine.predict("Team A", "Team B")
        self.assertEqual(pred["home_team"], "Team A")
        self.assertEqual(pred["away_team"], "Team B")
        self.assertIn("outcome_probabilities", pred)
        self.assertEqual(sum(pred["outcome_probabilities"].values()), 1.0)

    def test_serialization_roundtrip(self):
        matches = [
            {
                "status": "full_time",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 3,
                "away_score": 0,
                "date": "2026-08-01T15:00:00",
            }
        ]
        engine = PredictionEngine()
        engine.fit(matches)
        data = engine.to_dict()
        restored = PredictionEngine.from_dict(data)
        self.assertTrue(restored.fitted)
        self.assertEqual(restored.teams, engine.teams)
        self.assertEqual(restored.last_trained, engine.last_trained)


if __name__ == "__main__":
    unittest.main()
