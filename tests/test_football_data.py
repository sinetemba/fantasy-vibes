"""Tests for the football-data.org source."""

import os
import unittest
from unittest.mock import patch

os.environ["FOOTBALL_DATA_API_KEY"] = "test-key"

from src.data.sources.football_data import FootballDataSource


class TestFootballDataSource(unittest.TestCase):

    @patch("src.data.sources.football_data.cached_get")
    def test_api_get_builds_full_url_with_params(self, mock_cached_get):
        mock_cached_get.return_value = {"matches": []}
        source = FootballDataSource()
        result = source._api_get("/competitions/PL/matches", {"season": "2026"})
        self.assertEqual(result, {"matches": []})
        mock_cached_get.assert_called_once()
        call_args = mock_cached_get.call_args
        url = call_args[0][0]
        self.assertIn("/competitions/PL/matches", url)
        self.assertIn("season=2026", url)

    @patch("src.data.sources.football_data.cached_get")
    def test_get_matches_returns_empty_when_no_data(self, mock_cached_get):
        mock_cached_get.return_value = None
        source = FootballDataSource()
        matches = source.get_matches({"football_data": {"code": "PL"}}, season="2026")
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
