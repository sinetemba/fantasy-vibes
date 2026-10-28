"""Composite data source that cascades over multiple free providers."""

from typing import Any, Dict, List, Optional, Tuple

from .base import DataSource


class CompositeDataSource(DataSource):
    """
    Tries several data providers in order and returns the first non-empty
    result.  This allows the app to prefer live APIs (e.g. football-data.org)
    when configured, but fall back to public static datasets otherwise.
    """

    name = "composite"

    def __init__(
        self,
        match_sources: List[DataSource],
        standings_sources: List[DataSource],
    ):
        self.match_sources = match_sources
        self.standings_sources = standings_sources
        # Name of the source that produced the most recent result, so callers
        # don't have to re-query every source to work out where data came from.
        self.last_matches_source: Optional[str] = None
        self.last_standings_source: Optional[str] = None

    def is_available(self, league_config: Dict[str, Any]) -> bool:
        return any(s.is_available(league_config) for s in self.match_sources + self.standings_sources)

    def get_matches(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        matches, self.last_matches_source = self.get_matches_with_source(
            league_config, season
        )
        return matches

    def get_standings(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        standings, self.last_standings_source = self.get_standings_with_source(
            league_config, season
        )
        return standings

    # The *_with_source variants return the producing source alongside the
    # data — safe when one composite is shared across threads, unlike the
    # last_*_source attributes which any concurrent call can overwrite.
    def get_matches_with_source(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        for source in self.match_sources:
            if not source.is_available(league_config):
                continue
            matches = source.get_matches(league_config, season)
            if matches:
                return matches, source.name
        return [], None

    def get_standings_with_source(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        for source in self.standings_sources:
            if not source.is_available(league_config):
                continue
            standings = source.get_standings(league_config, season)
            if standings:
                return standings, source.name
        return [], None
