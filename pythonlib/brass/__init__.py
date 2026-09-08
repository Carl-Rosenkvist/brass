from ._brass import (
    BinaryReader,
    DileptonHistogrammer,
    EndBlock,
    Header,
    HistogramGroupBy,
    HistogramRequest,
    HistogramResult,
    HistogramRunner,
    HistogramTarget,
    IntegerAxis,
    InteractionBlock,
    ParticleBlock,
    Particles,
    RegularAxis,
    VariableAxis,
    particle_size_from_quantities,
)
from .analysis import Analysis
from .analysis_runner import AnalysisRunner
from .decays import DecayReconstructor
from .scan.scan import Scan


def histograms(reader, requests):
    runner = HistogramRunner(reader, requests)
    runner.run()
    return runner.result()


def histogram(reader, request):
    return histograms(reader, [request])[0]


__all__ = [
    "BinaryReader",
    "Header",
    "ParticleBlock",
    "EndBlock",
    "InteractionBlock",
    "Particles",
    "RegularAxis",
    "VariableAxis",
    "IntegerAxis",
    "HistogramResult",
    "HistogramRequest",
    "HistogramGroupBy",
    "HistogramTarget",
    "HistogramRunner",
    "DileptonHistogrammer",
    "histogram",
    "histograms",
    "particle_size_from_quantities",
    "DecayReconstructor",
]
