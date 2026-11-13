import numpy as np

from ..base import RunData, collect_runs

KINDS = ("xf", "xf_pt", "y", "y_mt")


class BaryonStoppingData(RunData):
    """Parameters and per-event x_F and rapidity histograms of BaryonStopping results.

    For each PDG and run: ``counts["xf"]`` (particles per x_F bin),
    ``counts["xf_pt"]`` (their summed p_T), ``counts["y"]`` (particles per
    rapidity bin) and ``counts["y_mt"]`` (their summed m_T), all per event
    and of shape (n_runs, n_bins). x_F bins are the same at every energy.
    """

    def __init__(self, data, parameter_paths):
        self.xf_edges = {}
        self.y_edges = {}
        self._per_run = {kind: {} for kind in KINDS}
        super().__init__(data, parameter_paths)

    def _add_run(self, run, analysis):
        histograms = getattr(analysis, "histograms", None)
        if not (isinstance(histograms, dict) and set(KINDS) <= set(histograms)):
            raise TypeError(
                f"run {run} ({analysis.folder_path}) is not a BaryonStopping result"
            )
        for kind in KINDS:
            for pdg, histo in histograms[kind].items():
                (edges,) = (np.asarray(e, dtype=float) for e in histo.edges)
                if kind.startswith("xf"):
                    edges = 2.0 * edges / analysis.sqrts
                    self._check_edges(self.xf_edges, pdg, edges, run, "x_F")
                else:
                    self._check_edges(self.y_edges, pdg, edges, run, "rapidity")
                values = np.asarray(histo.values, dtype=float) / analysis.events
                self._per_run[kind].setdefault(pdg, []).append(values)

    @staticmethod
    def _check_edges(stored, pdg, edges, run, what):
        if pdg not in stored:
            stored[pdg] = edges
        elif not np.allclose(edges, stored[pdg], rtol=0, atol=1e-9):
            raise ValueError(f"PDG {pdg}: {what} bins in run {run} differ from run 0")

    def _finish(self, n_runs):
        self.counts = {
            kind: collect_runs(self._per_run[kind], n_runs, "BaryonStopping results")
            for kind in KINDS
        }
        del self._per_run

    @property
    def pdgs(self):
        return list(self.counts["xf"])

    def _mean(self, pdg, kind):
        """Mean of the weight per bin, in the bins filled in every run."""
        n, total = self.counts[kind][pdg], self.counts[kind + "_" + _WEIGHT[kind]][pdg]
        filled = np.all(n > 0, axis=0)
        return filled, total[:, filled] / n[:, filled]

    def observable(self, pdg, name):
        """Training values of an observable, shape (n_runs, n_bins).

        ``"dNdxF"`` and ``"dNdy"`` are per event. ``"meanpT_xF"`` and
        ``"meanmT_y"`` are given only in the bins that hold particles in
        every run (see :meth:`axis`).
        """
        if name == "dNdxF":
            return self.counts["xf"][pdg] / np.diff(self.xf_edges[pdg])
        if name == "dNdy":
            return self.counts["y"][pdg] / np.diff(self.y_edges[pdg])
        if name == "meanpT_xF":
            return self._mean(pdg, "xf")[1]
        if name == "meanmT_y":
            return self._mean(pdg, "y")[1]
        raise ValueError(f"unknown observable {name!r}")

    def axis(self, pdg, name):
        """Bin centres of an observable."""
        kind = {"dNdxF": "xf", "meanpT_xF": "xf", "dNdy": "y", "meanmT_y": "y"}[name]
        edges = self.xf_edges[pdg] if kind == "xf" else self.y_edges[pdg]
        centres = 0.5 * (edges[:-1] + edges[1:])
        if name.startswith("mean"):
            return centres[self._mean(pdg, kind)[0]]
        return centres


_WEIGHT = {"xf": "pt", "y": "mt"}
