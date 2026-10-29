from brass.analysis import Analysis

from .bulk import Bulk
from .centrality import Centrality
from .dileptons import Dileptons

ANALYSES: dict[str, type[Analysis]] = {
    "dileptons": Dileptons,
    "bulk": Bulk,
    "centrality": Centrality,
}

__all__ = [
    "ANALYSES",
    "Bulk",
    "Dileptons",
    "Centrality",
]
