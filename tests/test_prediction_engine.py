"""Basic tests for the prediction engine."""

import unittest

from src.data.prediction_engine import (
    DRAW_FLOOR,
    PredictionEngine,
    _draw_prob,
)


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

    def test_duplicate_fixtures_deduped(self):
        """The same fixture arriving via two feeds is only counted once —
        the training pool mixes current, historical and related pulls."""
        base = {
            "status": "full_time",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_score": 2,
            "away_score": 1,
            "date": "2026-08-01T15:00:00",
        }
        # Same fixture, different feed spelling and kickoff drift.
        dup = dict(base, home_team="Team A FC", away_team="Team B FC",
                   date="2026-08-01T21:30:00")
        distinct = dict(base, date="2026-08-05T15:00:00")
        engine = PredictionEngine()
        engine.fit([base, dup, distinct])
        self.assertEqual(engine.matches_used, 2)

    def test_data_freshness_metadata(self):
        """fit() records how much data trained the model and through when."""
        matches = [
            {
                "status": "full_time",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 1,
                "away_score": 0,
                "date": "2026-08-01T15:00:00",
                "competition": "PL",
            }
        ]
        engine = PredictionEngine()
        engine.fit(matches, primary_competition="PL")
        self.assertEqual(engine.matches_used, 1)
        self.assertIsNotNone(engine.data_as_of)
        pred = engine.predict("Home", "Away")
        self.assertEqual(pred["matches_used"], 1)
        self.assertIsNotNone(pred["data_as_of"])
        restored = PredictionEngine.from_dict(engine.to_dict())
        self.assertEqual(restored.data_as_of, engine.data_as_of)

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

    def test_resolve_name_uses_aliases_and_equivalence(self):
        """Feed spellings like "Inter" resolve to the canonical model name
        instead of silently degrading to the fallback path."""
        matches = [
            {
                "status": "full_time",
                "home_team": "FC Internazionale Milano",
                "away_team": "SSC Napoli",
                "home_score": 2,
                "away_score": 1,
                "date": "2026-08-01T15:00:00",
            },
        ]
        engine = PredictionEngine()
        engine.fit(matches)
        self.assertEqual(engine.resolve_name("Inter"), "FC Internazionale Milano")
        self.assertEqual(engine.resolve_name("Napoli"), "SSC Napoli")
        pred = engine.predict("Inter", "SSC Napoli")
        self.assertEqual(pred["model"], "trained_poisson_elo")

    def test_competition_specific_baselines(self):
        """Scoring baselines are kept per competition — a high-scoring
        tournament doesn't share one average with a low-scoring one."""
        matches = []
        for i in range(40):
            matches.append({
                "status": "full_time",
                "home_team": f"A{i % 10}",
                "away_team": f"B{(i + 3) % 10}",
                "home_score": 3,
                "away_score": 2,
                "date": f"2026-08-{(i % 28) + 1:02d}T15:00:00",
                "competition": "HIGH",
            })
            # Different month so the two fixtures aren't merged as duplicate
            # feed rows — the same pair can't realistically play twice in a day.
            matches.append({
                "status": "full_time",
                "home_team": f"A{i % 10}",
                "away_team": f"B{(i + 3) % 10}",
                "home_score": 0,
                "away_score": 1,
                "date": f"2026-09-{(i % 28) + 1:02d}T16:00:00",
                "competition": "LOW",
            })
        engine = PredictionEngine()
        engine.fit(matches, primary_competition="LOW")
        self.assertIn("HIGH", engine.comp_avg)
        self.assertIn("LOW", engine.comp_avg)
        self.assertGreater(
            engine.comp_avg["HIGH"][0] + engine.comp_avg["HIGH"][1],
            engine.comp_avg["LOW"][0] + engine.comp_avg["LOW"][1],
        )

    def test_serialization_roundtrip_new_fields(self):
        matches = [
            {
                "status": "full_time",
                "home_team": "Home",
                "away_team": "Away",
                "home_score": 3,
                "away_score": 0,
                "date": "2026-08-01T15:00:00",
                "competition": "PL",
            }
        ]
        engine = PredictionEngine()
        engine.fit(matches, primary_competition="PL")
        restored = PredictionEngine.from_dict(engine.to_dict())
        self.assertEqual(restored.dc_rho, engine.dc_rho)
        self.assertEqual(restored.comp_avg, engine.comp_avg)
        self.assertEqual(restored.elo_unrated, engine.elo_unrated)
        self.assertEqual(restored.elo_prior, engine.elo_prior)

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

    @staticmethod
    def _equal_strength_league():
        """Double round-robin of identical teams scoring a fixed 2-1 at
        home — any team-level skew in predictions is a model artefact."""
        teams = [f"Team {c}" for c in "ABCDEF"]
        matches = []
        day = 0
        for _ in range(2):
            for home in teams:
                for away in teams:
                    if home == away:
                        continue
                    day += 1
                    matches.append({
                        "status": "full_time", "home_team": home, "away_team": away,
                        "home_score": 2, "away_score": 1, "competition": "L",
                        "date": f"2026-{1 + day // 28:02d}-{1 + day % 28:02d}T15:00:00",
                    })
        return matches

    def test_home_advantage_not_double_counted(self):
        # The venue baselines already carry home advantage; expected goals
        # must match them, not be inflated by the home factor again.
        engine = PredictionEngine()
        engine.fit(self._equal_strength_league(), primary_competition="L")
        pred = engine.predict("Team A", "Team B", competition="L")
        self.assertAlmostEqual(pred["expected_goals"]["home"], 2.0, delta=0.1)
        self.assertAlmostEqual(pred["expected_goals"]["away"], 1.0, delta=0.1)

    def test_neutral_venue_is_symmetric(self):
        # At a neutral venue the listed "home" side gets no edge, so swapping
        # the sides must swap the numbers.
        engine = PredictionEngine()
        engine.fit(self._equal_strength_league(), primary_competition="L")
        ab = engine.predict("Team A", "Team B", neutral=True, competition="L")
        ba = engine.predict("Team B", "Team A", neutral=True, competition="L")
        self.assertAlmostEqual(ab["expected_goals"]["home"], ba["expected_goals"]["away"], delta=0.05)
        self.assertAlmostEqual(ab["expected_goals"]["away"], ba["expected_goals"]["home"], delta=0.05)
        self.assertAlmostEqual(
            ab["outcome_probabilities"]["home_win"],
            ba["outcome_probabilities"]["away_win"],
            delta=0.02,
        )

    def test_draw_prob_scales_below_floor(self):
        """A heavy mismatch should price the draw near the (low) floor,
        not at a flat ~20% — tight games keep the learned rate."""
        self.assertAlmostEqual(_draw_prob(0, 0.25), 0.25)
        self.assertAlmostEqual(_draw_prob(1100, 0.30), DRAW_FLOOR)
        self.assertLess(_draw_prob(1100, 0.30), 0.11)

    def test_form_factor_measures_delta_vs_season(self):
        """Form at the season baseline → no adjustment; a hot run only
        counts the gap, since table PPG is already priced by context."""
        engine = PredictionEngine()
        engine.form_ppg = {"Team A": 0.8}  # 2.4 pts/game recently
        engine.comp_form_ppg = {}
        self.assertAlmostEqual(engine._form_factor("Team A", ctx={"ppg": 2.4}), 1.0)
        self.assertGreater(engine._form_factor("Team A", ctx={"ppg": 1.2}), 1.0)
        # No context → old behaviour (baseline is the 1.5 pts/game average).
        self.assertGreater(engine._form_factor("Team A"), 1.0)

    def test_naive_dates_parsed_as_sast(self):
        """Feeds store naive SAST kickoffs — fit() converts to true UTC,
        so the newest-result stamp sits 2h behind the raw string."""
        engine = PredictionEngine()
        engine.fit([{
            "status": "full_time", "home_team": "H", "away_team": "A",
            "home_score": 1, "away_score": 0, "date": "2026-08-01T15:00:00",
        }])
        self.assertEqual(engine.data_as_of, "2026-08-01T13:00:00+00:00")

    def test_ambiguous_prefix_names_stay_separate(self):
        engine = PredictionEngine()
        names = engine._canonical_map(
            ["Inter", "Inter Miami", "Internazionale", "Real", "Real Madrid",
             "Real Betis", "Tottenham", "Tottenham Hotspur"]
        )
        self.assertNotEqual(names["Inter Miami"], names["Inter"])
        self.assertEqual(names["Internazionale"], names["Inter"])
        self.assertNotEqual(names["Real Madrid"], names["Real Betis"])
        self.assertEqual(names["Tottenham Hotspur"], names["Tottenham"])


class TestFallback(unittest.TestCase):
    """The rating-only path used when a team isn't in the training pool —
    it should still respect home field, Elo-gap-scaled draws and any
    fitted ratings the known side has."""

    def _fitted(self):
        # Mixed results so the learned draw rate (~40%, clamped to 0.30)
        # sits well above DRAW_FLOOR — otherwise the floor hides the
        # Elo-gap scaling this suite checks.
        matches = []
        i = 0
        for (home, away, hs, aws), n in [
            (("Team A", "Team B", 5, 0), 8),
            (("Team B", "Team A", 1, 0), 4),
            (("Team A", "Team B", 1, 1), 4),
            (("Team B", "Team A", 2, 2), 4),
        ]:
            for _ in range(n):
                i += 1
                matches.append({
                    "status": "full_time", "home_team": home, "away_team": away,
                    "home_score": hs, "away_score": aws, "competition": "L",
                    "date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}T15:00:00",
                })
        engine = PredictionEngine()
        engine.fit(matches, initial_elo={"Giant": 1900, "Minnow": 800}, primary_competition="L")
        return engine

    def test_home_advantage_applies(self):
        engine = self._fitted()
        pred = engine.predict("Ghost FC", "Phantom FC")  # both unknown → fallback
        self.assertEqual(pred["model"], "elo_fallback")
        self.assertGreater(
            pred["outcome_probabilities"]["home_win"],
            pred["outcome_probabilities"]["away_win"],
        )

    def test_neutral_removes_home_edge(self):
        engine = self._fitted()
        pred = engine.predict("Ghost FC", "Phantom FC", neutral=True)
        self.assertAlmostEqual(
            pred["outcome_probabilities"]["home_win"],
            pred["outcome_probabilities"]["away_win"],
        )

    def test_draw_shrinks_with_elo_gap(self):
        engine = self._fitted()
        even = engine.predict("Ghost FC", "Phantom FC", neutral=True)
        mismatch = engine.predict("Giant", "Minnow", neutral=True)
        self.assertLess(
            mismatch["outcome_probabilities"]["draw"],
            even["outcome_probabilities"]["draw"],
        )

    def test_known_side_keeps_fitted_ratings(self):
        # A strong fitted home attack must flow into expected goals even
        # when the opponent resolves to nothing — not a league-average guess.
        engine = self._fitted()
        engine.home_attack["Team A"] = 2.0
        pred = engine.predict("Team A", "Ghost FC")
        self.assertEqual(pred["model"], "elo_fallback")
        self.assertAlmostEqual(
            pred["expected_goals"]["home"],
            min(5.0, engine.league_avg_home * 2.0),
            delta=0.01,
        )


if __name__ == "__main__":
    unittest.main()
