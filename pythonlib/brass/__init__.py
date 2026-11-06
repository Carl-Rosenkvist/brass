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
import importlib

# Loaded on first use, so that `import brass` stays fast for scripts that
# only read files and fill histograms (the analysis framework pulls in
# PyYAML and multiprocessing).
_LAZY = {
    "Analysis": ".analysis",
    "AnalysisRunner": ".analysis_runner",
    "DecayReconstructor": ".decays",
    "Scan": ".scan.scan",
}


def __getattr__(name):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


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
