from brass.analysis import Analysis

from .baryon_stopping import BaryonStopping
from .bulk import Bulk
from .bulk_observables import BulkObservables
from .centrality import Centrality
from .collision_counts import CollisionCounts
from .dileptons import Dileptons
from .multiplicity_vs_time import MultiplicityVsTime
from .reaction_counts import ReactionCounts

ANALYSES: dict[str, type[Analysis]] = {
    "dileptons": Dileptons,
    "bulk": Bulk,
    "bulk-observables": BulkObservables,
    "baryon-stopping": BaryonStopping,
    "centrality": Centrality,
    "collision-counts": CollisionCounts,
    "reaction-counts": ReactionCounts,
    "multiplicity-vs-time": MultiplicityVsTime,
}

__all__ = [
    "ANALYSES",
    "BaryonStopping",
    "Bulk",
    "BulkObservables",
    "Dileptons",
    "MultiplicityVsTime",
    "ReactionCounts",
    "Centrality",
    "CollisionCounts",
]
