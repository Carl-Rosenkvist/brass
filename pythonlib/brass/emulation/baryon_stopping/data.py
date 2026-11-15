from itertools import pairwise

import numpy as np

from ..base import RunData, collect_runs

KINDS = ("xf", "xf_pt", "y", "y_mt")


class BaryonStoppingData(RunData):
    """Parameters and per-event x_F and rapidity histograms of BaryonStopping results.

    For each PDG and run: ``counts["xf"]`` (particles per x_F bin),
    ``counts["xf_pt"]`` (their summed p_T), ``counts["y"]`` (particles per
    rapidity bin) and ``counts["y_mt"]`` (their summed m_T), all per event
    and of shape (n_runs, n_bins). x_F bins are the same at every energy.
    ``events`` holds the number of events of each run.
    """

    def __init__(self, data, parameter_paths):
        self.xf_edges = {}
        self.y_edges = {}
        self._per_run = {kind: {} for kind in KINDS}
        self._events = []
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
        self._events.append(analysis.events)

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
        self.events = np.asarray(self._events, dtype=float)
        del self._per_run, self._events

    @property
    def pdgs(self):
        return list(self.counts["xf"])

    def _spectrum(self, pdg, name, xf_group=1, xf_max=None, fold=False, min_count=None):
        """(bin edges, values of shape (n_runs, n_bins)) of an observable.

        The binning is changed in this order: merge ``xf_group`` x_F bins,
        fold -x and +x, keep |x_F| <= ``xf_max``, merge to ``min_count``.
        """
        if name not in _KIND:
            raise ValueError(f"unknown observable {name!r}")
        kind = _KIND[name]
        n = self.counts[kind][pdg]  # particles per event and bin
        total = self.counts[kind + "_" + _WEIGHT[kind]][pdg]  # their summed weight
        edges = self.xf_edges[pdg] if kind == "xf" else self.y_edges[pdg]
        settings = (
            f"xf_group={xf_group}, xf_max={xf_max}, fold={fold}, min_count={min_count}"
        )

        if kind == "xf":
            xf_group = int(xf_group)
            if xf_group < 1 or n.shape[1] % xf_group:
                raise ValueError(
                    f"xf_group={xf_group} does not divide the {n.shape[1]} x_F bins"
                )
            n = n.reshape(len(n), -1, xf_group).sum(axis=2)
            total = total.reshape(len(total), -1, xf_group).sum(axis=2)
            edges = edges[::xf_group]

        if fold:
            edges, n, total = _fold(edges, n, total)

        if kind == "xf" and xf_max is not None:
            tol = 1e-9
            keep = (edges[:-1] >= -xf_max - tol) & (edges[1:] <= xf_max + tol)
            if not keep.any():
                raise ValueError(f"PDG {pdg}: no usable bins for {name!r} ({settings})")
            first, last = np.flatnonzero(keep)[[0, -1]]
            edges, n, total = (
                edges[first : last + 2],
                n[:, first : last + 1],
                total[:, first : last + 1],
            )

        if min_count is not None:
            if min_count <= 0:
                raise ValueError("min_count must be positive")
            edges, n, total = _merge(
                edges, n, total, n * self.events[:, None], min_count
            )

        width = np.diff(edges) * (2.0 if fold else 1.0)
        keep = np.ones(n.shape[1], dtype=bool)
        if name.startswith("mean"):
            keep &= np.all(n > 0, axis=0)
        if not keep.any():
            raise ValueError(
                f"PDG {pdg}: no usable bins for {name!r} ({settings}); "
                "mean values need particles in every run"
            )
        lo, hi = edges[:-1][keep], edges[1:][keep]
        if name.startswith("mean"):
            return (lo, hi), total[:, keep] / n[:, keep]
        return (lo, hi), n[:, keep] / width[keep]

    def observable(
        self, pdg, name, xf_group=1, xf_max=None, fold=False, min_count=None
    ):
        """Training values of an observable, shape (n_runs, n_bins).

        ``"dNdxF"`` and ``"dNdy"`` are per event. ``"meanpT_xF"`` and
        ``"meanmT_y"`` are given only in the bins that hold particles in
        every run. The binning can be changed, in this order (counts and
        summed pT or mT add exactly):

        ``xf_group``
            merge that many neighbouring x_F bins;
        ``fold``
            add the bins at -x and +x (symmetric systems such as pp); the
            observables are then given for x >= 0, and dN/dx is the average
            of both sides;
        ``xf_max``
            keep only x_F bins within |x_F| <= xf_max;
        ``min_count``
            merge neighbouring bins, outward from x = 0, until each holds at
            least this many particles in every run. Outer bins without any
            particles are dropped; a last bin that stays below the minimum
            joins its inner neighbour.
        """
        return self._spectrum(pdg, name, xf_group, xf_max, fold, min_count)[1]

    def bins(self, pdg, name, xf_group=1, xf_max=None, fold=False, min_count=None):
        """Lower and upper bin edges of an observable (see :meth:`observable`)."""
        return self._spectrum(pdg, name, xf_group, xf_max, fold, min_count)[0]

    def axis(self, pdg, name, xf_group=1, xf_max=None, fold=False, min_count=None):
        """Bin centres of an observable (see :meth:`observable`)."""
        lo, hi = self.bins(pdg, name, xf_group, xf_max, fold, min_count)
        return 0.5 * (lo + hi)


def _fold(edges, n, total):
    """Add the bins at -x and +x; returns the bins for x >= 0."""
    if not np.allclose(edges, -edges[::-1], rtol=0, atol=1e-9):
        raise ValueError("fold needs bins symmetric around 0")
    half = n.shape[1] // 2
    if n.shape[1] % 2:  # middle bin straddles 0: keep all of it, half its width
        n_f = np.concatenate(
            [n[:, half : half + 1], n[:, half + 1 :] + n[:, :half][:, ::-1]], 1
        )
        t_f = np.concatenate(
            [
                total[:, half : half + 1],
                total[:, half + 1 :] + total[:, :half][:, ::-1],
            ],
            1,
        )
        return np.concatenate([[0.0], edges[half + 1 :]]), n_f, t_f
    return (
        edges[half:],
        n[:, half:] + n[:, :half][:, ::-1],
        total[:, half:] + total[:, :half][:, ::-1],
    )


def _merge(edges, n, total, particles, min_count):
    """Merge neighbouring bins outward from x = 0 to >= min_count particles per run."""
    n_bins = n.shape[1]
    inner = np.flatnonzero(np.isclose(edges, 0.0, rtol=0, atol=1e-9))
    start = int(inner[0]) if len(inner) else 0
    sides = [list(range(start, n_bins)), list(range(start - 1, -1, -1))]

    groups = []
    for order in sides:
        filled = [i for i in order if particles[:, i].sum() > 0]
        if not filled:
            continue
        order = order[: order.index(filled[-1]) + 1]  # drop empty outer bins
        side, current = [], []
        for i in order:
            current.append(i)
            if particles[:, current].sum(axis=1).min() >= min_count:
                side.append(current)
                current = []
        if current:
            if side:
                side[-1] = side[-1] + current
            else:
                side.append(current)
        groups.extend(sorted(g) for g in side)

    groups.sort()
    new_edges = [edges[g[0]] for g in groups] + [edges[groups[-1][-1] + 1]]
    for a, b in pairwise(groups):  # groups must be contiguous
        if a[-1] + 1 != b[0]:
            raise ValueError("empty bins between filled ones; use a smaller range")
    return (
        np.asarray(new_edges),
        np.stack([n[:, g].sum(axis=1) for g in groups], axis=1),
        np.stack([total[:, g].sum(axis=1) for g in groups], axis=1),
    )


_KIND = {"dNdxF": "xf", "meanpT_xF": "xf", "dNdy": "y", "meanmT_y": "y"}
_WEIGHT = {"xf": "pt", "y": "mt"}
