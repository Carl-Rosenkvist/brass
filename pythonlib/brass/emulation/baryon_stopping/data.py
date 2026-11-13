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

    def _spectrum(self, pdg, name, xf_group=1, xf_max=None):
        """(bin centres, values of shape (n_runs, n_bins)) of an observable."""
        if name not in _KIND:
            raise ValueError(f"unknown observable {name!r}")
        kind = _KIND[name]
        n = self.counts[kind][pdg]
        total = self.counts[kind + "_" + _WEIGHT[kind]][pdg]
        edges = self.xf_edges[pdg] if kind == "xf" else self.y_edges[pdg]

        if kind == "xf":
            xf_group = int(xf_group)
            if xf_group < 1 or n.shape[1] % xf_group:
                raise ValueError(
                    f"xf_group={xf_group} does not divide the {n.shape[1]} x_F bins"
                )
            n = n.reshape(len(n), -1, xf_group).sum(axis=2)
            total = total.reshape(len(total), -1, xf_group).sum(axis=2)
            edges = edges[::xf_group]
            keep = np.ones(n.shape[1], dtype=bool)
            if xf_max is not None:
                tol = 1e-9
                keep = (edges[:-1] >= -xf_max - tol) & (edges[1:] <= xf_max + tol)
        else:
            keep = np.ones(n.shape[1], dtype=bool)

        if name.startswith("mean"):
            keep &= np.all(n > 0, axis=0)
        if not keep.any():
            raise ValueError(
                f"PDG {pdg}: no usable bins for {name!r} (xf_group={xf_group}, "
                f"xf_max={xf_max}); mean values need particles in every run"
            )

        centres = 0.5 * (edges[:-1] + edges[1:])[keep]
        if name.startswith("mean"):
            return centres, total[:, keep] / n[:, keep]
        return centres, n[:, keep] / np.diff(edges)[keep]

    def observable(self, pdg, name, xf_group=1, xf_max=None):
        """Training values of an observable, shape (n_runs, n_bins).

        ``"dNdxF"`` and ``"dNdy"`` are per event. ``"meanpT_xF"`` and
        ``"meanmT_y"`` are given only in the bins that hold particles in
        every run. For the x_F observables, ``xf_group`` merges that many
        neighbouring x_F bins (counts and summed pT add exactly) and
        ``xf_max`` keeps only bins within |x_F| <= xf_max.
        """
        return self._spectrum(pdg, name, xf_group, xf_max)[1]

    def axis(self, pdg, name, xf_group=1, xf_max=None):
        """Bin centres of an observable (see :meth:`observable`)."""
        return self._spectrum(pdg, name, xf_group, xf_max)[0]


_KIND = {"dNdxF": "xf", "meanpT_xF": "xf", "dNdy": "y", "meanmT_y": "y"}
_WEIGHT = {"xf": "pt", "y": "mt"}
