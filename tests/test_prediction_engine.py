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
        self.assertEqual(restored.comp_form_ppg, engine.comp_form_ppg)
        self.assertEqual(restored.name_index, engine.name_index)

    def test_cross_source_name_canonicalisation(self):
        """The same club named differently across sources is merged."""
        matches = [
            {
                "status": "full_time",
                "home_team": "Arsenal FC",
                "away_team": "Chelsea FC",
                "home_score": 2,
                "away_score": 0,
                "date": "2026-08-01T15:00:00",
                "competition": "PL",
            },
            {
                "status": "full_time",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_score": 1,
                "away_score": 1,
                "date": "2026-08-08T15:00:00",
                "competition": "CL",
            },
        ]
        engine = PredictionEngine()
        engine.fit(matches)
        self.assertEqual(len(engine.teams), 2)
        canonical = engine.resolve_name("Arsenal FC")
        self.assertEqual(canonical, engine.resolve_name("Arsenal"))
        self.assertIn(canonical, engine.teams)

    def test_competition_specific_form(self):
        """Form is tracked per competition as well as overall."""
        matches = []
        for i in range(6):
            matches.append({
                "status": "full_time",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 3,
                "away_score": 0,
                "date": f"2026-08-0{i + 1}T15:00:00",
                "competition": "PL",
            })
        matches.append({
            "status": "full_time",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_score": 0,
            "away_score": 3,
            "date": "2026-08-10T15:00:00",
            "competition": "CL",
        })
        engine = PredictionEngine()
        engine.fit(matches)
        self.assertIn("CL|Team A", engine.comp_form_ppg)
        self.assertIn("PL|Team A", engine.comp_form_ppg)
        # Team A lost its only CL match but won all PL matches.
        self.assertLess(engine.comp_form_ppg["CL|Team A"], engine.comp_form_ppg["PL|Team A"])

    def test_context_factor_position(self):
        """Top-of-log teams get a boost, bottom teams a penalty."""
        top = PredictionEngine._context_factor({"position": 1, "league_size": 20, "ppg": 2.5})
        bottom = PredictionEngine._context_factor({"position": 20, "league_size": 20, "ppg": 0.5})
        self.assertGreater(top, 1.0)
        self.assertLess(bottom, 1.0)
        self.assertEqual(PredictionEngine._context_factor(None), 1.0)
        self.assertEqual(PredictionEngine._context_factor({}), 1.0)

    def test_predict_uses_contexts(self):
        """A prediction incorporates log-position context and reports inputs."""
        matches = [
            {
                "status": "full_time",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 1,
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
        contexts = {
            "Team A": {"position": 1, "league_size": 20, "ppg": 2.5, "competition": "PL"},
            "Team B": {"position": 20, "league_size": 20, "ppg": 0.5, "competition": "PL"},
        }
        pred = engine.predict("Team A", "Team B", competition="PL", contexts=contexts)
        inputs = pred["model_inputs"]
        self.assertEqual(inputs["home"]["position"], 1)
        self.assertEqual(inputs["away"]["position"], 20)
        self.assertIsNotNone(inputs["home"]["form_ppg"])
        # Strong context should shift expected goals in Team A's favour.
        self.assertGreater(
            pred["expected_goals"]["home"], pred["expected_goals"]["away"]
        )


if __name__ == "__main__":
    unittest.main()
