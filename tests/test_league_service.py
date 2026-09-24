"""Tests for LeagueService helpers and composite source tracking."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
