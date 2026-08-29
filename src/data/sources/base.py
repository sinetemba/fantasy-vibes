"""Base data source interface for league data providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DataSource(ABC):
    """Abstract base class for a football data provider."""

    name: str = "base"

    @abstractmethod
    def is_available(self, league_config: Dict[str, Any]) -> bool:
        """Return True if this source can be used for the given league."""
        ...

    @abstractmethod
    def get_matches(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return a list of normalized match dicts for the league/season."""
        ...

    @abstractmethod
    def get_standings(
        self, league_config: Dict[str, Any], season: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return a list of normalized standing dicts for the league/season."""
        ...
