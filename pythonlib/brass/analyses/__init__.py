from brass.analysis import Analysis

from .angular_distributions import AngularDistributions
from .baryon_stopping import BaryonStopping
from .bulk import Bulk
from .bulk_observables import BulkObservables
from .centrality import Centrality
from .collision_counts import CollisionCounts
from .cross_sections import CrossSections
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
    "angular-distributions": AngularDistributions,
    "cross-sections": CrossSections,
    "multiplicity-vs-time": MultiplicityVsTime,
}

__all__ = [
    "ANALYSES",
    "AngularDistributions",
    "BaryonStopping",
    "Bulk",
    "BulkObservables",
    "Dileptons",
    "MultiplicityVsTime",
    "ReactionCounts",
    "Centrality",
    "CollisionCounts",
    "CrossSections",
]
