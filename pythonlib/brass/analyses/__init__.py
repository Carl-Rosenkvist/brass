from brass.analysis import Analysis

from .baryon_stopping import BaryonStopping
from .bulk import Bulk
from .centrality import Centrality
from .dileptons import Dileptons

ANALYSES: dict[str, type[Analysis]] = {
    "dileptons": Dileptons,
    "bulk": Bulk,
    "baryon-stopping": BaryonStopping,
    "centrality": Centrality,
}

__all__ = [
    "ANALYSES",
    "BaryonStopping",
    "Bulk",
    "Dileptons",
    "Centrality",
]
