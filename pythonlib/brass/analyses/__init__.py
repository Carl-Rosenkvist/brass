from brass.analysis import Analysis

from .baryon_stopping import BaryonStopping
from .bulk import Bulk
from .bulk_observables import BulkObservables
from .centrality import Centrality
from .collision_counts import CollisionCounts
from .dileptons import Dileptons

ANALYSES: dict[str, type[Analysis]] = {
    "dileptons": Dileptons,
    "bulk": Bulk,
    "bulk-observables": BulkObservables,
    "baryon-stopping": BaryonStopping,
    "centrality": Centrality,
    "collision-counts": CollisionCounts,
}

__all__ = [
    "ANALYSES",
    "BaryonStopping",
    "Bulk",
    "BulkObservables",
    "Dileptons",
    "Centrality",
    "CollisionCounts",
]
