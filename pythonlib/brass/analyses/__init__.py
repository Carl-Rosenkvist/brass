from brass.analysis import Analysis

from .bulk import Bulk
from .dileptons import Dileptons

ANALYSES: dict[str, type[Analysis]] = {
    "dileptons": Dileptons,
    "bulk": Bulk,
}

__all__ = [
    "ANALYSES",
    "Bulk",
    "Dileptons",
]
