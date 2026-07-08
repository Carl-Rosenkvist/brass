from ._brass import (
    BinaryReader,
    Header,
    ParticleBlock,
    EndBlock,
    InteractionBlock,
    Particles,
    RegularAxis,
    VariableAxis,
    IntegerAxis,
    HistogramResult,
    HistogramRequest,
    HistogramGroupBy,
    HistogramTarget,
    HistogramRunner,
    DileptonHistogrammer,
    particle_size_from_quantities,
)

from .decays import DecayReconstructor


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
