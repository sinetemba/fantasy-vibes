"""Tests for LeagueService helpers."""

import unittest

from src.data.league_service import LeagueService


class DummySource:
    def __init__(self, name: str, data=None):
        self.name = name
        self._data = data if data is not None else []

    def is_available(self, league_config):
        return self._data is not None

    def get_matches(self, league_config):
        return self._data if getattr(self, "method", "") == "matches" else []

    def get_standings(self, league_config):
        return self._data if getattr(self, "method", "") == "standings" else []


class TestLeagueService(unittest.TestCase):

    def test_detect_source_picks_first_available(self):
        class Source1:
            name = "source1"
            def is_available(self, cfg):
                return True
            def get_matches(self, cfg):
                return []

        class Source2:
            name = "source2"
            def is_available(self, cfg):
                return True
            def get_matches(self, cfg):
                return [{"match_id": "1"}]

        ls = LeagueService.__new__(LeagueService)
        ls._current_league = {}
        result = ls._detect_source([Source1(), Source2()], "matches")
        self.assertEqual(result, "source2")

    def test_detect_source_returns_none_when_all_empty(self):
        class Source:
            name = "empty"
            def is_available(self, cfg):
                return True
            def get_matches(self, cfg):
                return []

        ls = LeagueService.__new__(LeagueService)
        ls._current_league = {}
        result = ls._detect_source([Source()], "matches")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
