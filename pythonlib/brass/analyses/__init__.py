from brass.analysis import Analysis

from .dileptons import Dileptons

ANALYSES: dict[str, type[Analysis]] = {
    "dileptons": Dileptons,
}


__all__ = [
    "ANALYSES",
    "Dileptons",
]
