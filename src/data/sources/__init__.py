from .base import DataSource
from .openfootball import OpenfootballSource
from .football_data import FootballDataSource
from .thesportsdb import TheSportsDBSource
from .composite import CompositeDataSource

__all__ = [
    "DataSource",
    "OpenfootballSource",
    "FootballDataSource",
    "TheSportsDBSource",
    "CompositeDataSource",
]
