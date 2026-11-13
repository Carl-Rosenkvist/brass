import copy
import pickle

import numpy as np

from .observables import SPECTRA, XF_OBSERVABLES, Observables


class DataHandler:
    """Parameters and normalised (y, mT) histograms from Bulk results.

    ``histograms[pdg]`` has shape (n_runs, n_y, n_mt) and holds
    d²N/(dy dmT) per event. If the Bulk analysis ran with ``--xf-bins``,
    ``xf_histograms[pdg]`` has shape (n_runs, n_xf, n_pt) and holds
    d²N/(dx_F dp_T) per event, with bin edges ``xf_edges`` and ``pt_edges``.
    """

    def __init__(self, data, parameter_paths):
        self.parameter_paths = dict(parameter_paths)
        self.parameter_names = list(self.parameter_paths)

        self.y_edges = {}
        self.mt_edges = {}
        self.xf_edges = {}
        self.pt_edges = {}
        self.xf_histograms = {}

        self.parameters, self.histograms = self._process(data)

    @classmethod
    def from_file(cls, path, parameter_paths):
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(data=data, parameter_paths=parameter_paths)

    @staticmethod
    def _get_nested(mapping, path):
        for key in path:
            if isinstance(mapping, list) and isinstance(key, str):
                mapping = _pythia_setting(mapping, key)
            else:
                mapping = mapping[key]

        return float(mapping)

    def _extract_parameters(self, analysis):
        values = []
        for name in self.parameter_names:
            try:
                values.append(
                    self._get_nested(analysis.config, self.parameter_paths[name])
                )
            except (KeyError, IndexError, TypeError) as error:
                raise KeyError(
                    f"parameter {name!r} not found at "
                    f"{self.parameter_paths[name]} in {analysis.folder_path}"
                ) from error
        return values

    def _process(self, data):
        parameters = []
        histograms = {}

        for run, result in enumerate(data):
            analysis = result["result"]

            if analysis.events <= 0:
                raise ValueError(f"run {run} ({analysis.folder_path}) has no events")

            parameters.append(self._extract_parameters(analysis))

            for pdg, histo in analysis.histograms.items():
                counts = np.asarray(histo.values, dtype=float)
                y_edges, mt_edges = (np.asarray(e, dtype=float) for e in histo.edges)

                if pdg not in self.y_edges:
                    if run > 0:
                        raise ValueError(
                            f"PDG {pdg} is missing in the runs before run {run}"
                        )
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
                histograms.setdefault(pdg, []).append(
                    counts / (analysis.events * dy[:, None] * dmt[None, :])
                )

            self._add_xf(run, analysis)

        n_runs = len(parameters)
        missing = {pdg: n_runs - len(h) for pdg, h in histograms.items() if len(h) != n_runs}
        if missing:
            raise ValueError(f"PDGs missing in some runs (pdg: runs missing): {missing}")

        with_xf = [len(h) for h in self.xf_histograms.values()]
        if self.xf_histograms and (set(self.xf_histograms) != set(histograms)
                                   or set(with_xf) != {n_runs}):
            raise ValueError("x_F histograms (--xf-bins) are missing in some runs")
        self.xf_histograms = {
            pdg: np.stack(h) for pdg, h in self.xf_histograms.items()
        }

        return (
            np.asarray(parameters, dtype=float),
            {pdg: np.stack(h) for pdg, h in histograms.items()},
        )

    def _add_xf(self, run, analysis):
        """Store d²N/(dx_F dp_T) of one run (Bulk with --xf-bins)."""
        xf_histograms = getattr(analysis, "xf_histograms", None)
        if not xf_histograms:
            return
        for pdg, histo in xf_histograms.items():
            pz_edges, pt_edges = (np.asarray(e, dtype=float) for e in histo.edges)
            xf_edges = 2.0 * pz_edges / analysis.sqrts
            if pdg not in self.xf_edges:
                self.xf_edges[pdg] = xf_edges
                self.pt_edges[pdg] = pt_edges
            elif not (
                np.allclose(xf_edges, self.xf_edges[pdg], rtol=0, atol=1e-9)
                and np.array_equal(pt_edges, self.pt_edges[pdg])
            ):
                raise ValueError(
                    f"PDG {pdg}: x_F histogram binning in run {run} differs from run 0"
                )
            counts = np.asarray(histo.values, dtype=float)
            self.xf_histograms.setdefault(pdg, []).append(
                counts
                / (
                    analysis.events
                    * np.diff(xf_edges)[:, None]
                    * np.diff(pt_edges)[None, :]
                )
            )

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
        if isinstance(name, XF_OBSERVABLES):
            if pdg not in self.xf_histograms:
                raise ValueError(
                    f"{name} needs x_F histograms: run the Bulk analysis "
                    "with --xf-bins"
                )
            return name.values(
                self.xf_histograms[pdg], self.xf_edges[pdg], self.pt_edges[pdg]
            )
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


def _pythia_setting(entries, key):
    """Value of ``key`` in a list of PYTHIA ``"key = value"`` strings."""
    for entry in entries:
        name, separator, value = entry.partition("=")
        if separator and name.strip() == key:
            return value.strip()
    raise KeyError(key)
