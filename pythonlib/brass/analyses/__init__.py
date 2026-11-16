from brass.analysis import Analysis

from .angular_distributions import AngularDistributions
from .baryon_stopping import BaryonStopping
from .bulk import Bulk
from .bulk_observables import BulkObservables
from .centrality import Centrality
from .collision_counts import CollisionCounts
from .cross_sections import CrossSections
from .dilepton_spectra import DileptonSpectra
from .dileptons import Dileptons
from .multiplicities import Multiplicities
from .multiplicity_vs_time import MultiplicityVsTime
from .position_grid import PositionGrid
from .reaction_counts import ReactionCounts
from .thermodynamics import Thermodynamics

ANALYSES: dict[str, type[Analysis]] = {
    "dileptons": Dileptons,
    "dilepton-spectra": DileptonSpectra,
    "bulk": Bulk,
    "bulk-observables": BulkObservables,
    "baryon-stopping": BaryonStopping,
    "centrality": Centrality,
    "collision-counts": CollisionCounts,
    "reaction-counts": ReactionCounts,
    "angular-distributions": AngularDistributions,
    "cross-sections": CrossSections,
    "multiplicity-vs-time": MultiplicityVsTime,
    "multiplicities": Multiplicities,
    "position-grid": PositionGrid,
    "thermodynamics": Thermodynamics,
}

__all__ = [
    "ANALYSES",
    "AngularDistributions",
    "BaryonStopping",
    "Bulk",
    "BulkObservables",
    "DileptonSpectra",
    "Dileptons",
    "Multiplicities",
    "MultiplicityVsTime",
    "PositionGrid",
    "ReactionCounts",
    "Thermodynamics",
    "Centrality",
    "CollisionCounts",
    "CrossSections",
]
