import copy

import numpy as np

from ..base import RunData, collect_runs
from .observables import SPECTRA, Observables


class DataHandler(RunData):
    """Parameters and normalised (y, mT) histograms from Bulk results.

    ``histograms[pdg]`` has shape (n_runs, n_y, n_mt) and holds
    d²N/(dy dmT) per event.
    """

    def __init__(self, data, parameter_paths):
        self.y_edges = {}
        self.mt_edges = {}
        self._per_run = {}
        super().__init__(data, parameter_paths)

    def _add_run(self, run, analysis):
        for pdg, histo in analysis.histograms.items():
            counts = np.asarray(histo.values, dtype=float)
            y_edges, mt_edges = (np.asarray(e, dtype=float) for e in histo.edges)

            if pdg not in self.y_edges:
                if run > 0:
                    raise ValueError(f"PDG {pdg} is missing in the runs before run {run}")
                self.y_edges[pdg] = y_edges
                self.mt_edges[pdg] = mt_edges
            elif not (
                np.array_equal(y_edges, self.y_edges[pdg])
                and np.array_equal(mt_edges, self.mt_edges[pdg])
            ):
                raise ValueError(
                    f"PDG {pdg}: histogram edges in run {run} differ from run 0"
                )

            dy = np.diff(y_edges)
            dmt = np.diff(mt_edges)
            self._per_run.setdefault(pdg, []).append(
                counts / (analysis.events * dy[:, None] * dmt[None, :])
            )

    def _finish(self, n_runs):
        self.histograms = collect_runs(self._per_run, n_runs, "Bulk results")
        del self._per_run

    @property
    def pdgs(self):
        return list(self.histograms)

    def observables(self, pdg):
        return Observables(self.y_edges[pdg], self.mt_edges[pdg])

    def observable(self, pdg, name):
        """Training values of an observable, shape (n_runs, n_outputs)."""
        H = self.histograms[pdg]
        obs = self.observables(pdg)
        if name == "hist2d":
            return H.reshape(len(H), -1)
        if name == "dNdy":
            return obs.dNdy(H)
        if name == "dNdmt":
            return obs.dNdmt(H)
        if isinstance(name, SPECTRA):
            _, A = name.build(obs)
            return H.reshape(len(H), -1) @ A.T
        raise ValueError(f"unknown observable {name!r}")

    def reduce(self, mt_max=None, mt_group=1):
        """Copy with fewer mT bins: crop at ``mt_max`` and merge ``mt_group`` bins.

        Bins whose upper edge lies above ``mt_max`` are dropped. Then each
        group of ``mt_group`` neighbouring bins is merged into one; the
        density d²N/(dy dmT) is averaged with the bin widths as weights, so
        the number of particles is unchanged. The rapidity axis is kept.

        Particles above ``mt_max`` are lost, also from dN/dy derived from the
        reduced histogram, so put the cut where the spectra have died out.
        """
        mt_group = int(mt_group)
        if mt_group < 1:
            raise ValueError("mt_group must be a positive integer")

        reduced = copy.copy(self)
        reduced.mt_edges = {}
        reduced.histograms = {}
        for pdg, H in self.histograms.items():
            edges = self.mt_edges[pdg]
            n = len(edges) - 1
            if mt_max is not None:
                n = int(np.sum(edges[1:] <= mt_max * (1 + 1e-12)))
                if n == 0:
                    raise ValueError(
                        f"PDG {pdg}: no mT bin below mt_max={mt_max} "
                        f"(first edges {edges[:2]})"
                    )
            if n % mt_group:
                raise ValueError(
                    f"PDG {pdg}: {n} mT bins cannot be merged in groups of "
                    f"{mt_group}; change mt_max or mt_group"
                )

            widths = np.diff(edges[: n + 1])
            counts = H[..., :n] * widths  # d²N/dy per mT bin
            counts = counts.reshape(*H.shape[:-1], n // mt_group, mt_group).sum(-1)
            new_edges = edges[: n + 1 : mt_group]
            reduced.mt_edges[pdg] = new_edges
            reduced.histograms[pdg] = counts / np.diff(new_edges)
        reduced.y_edges = dict(self.y_edges)
        return reduced

