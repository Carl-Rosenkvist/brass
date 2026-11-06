import pickle

import numpy as np

from .observables import Observables


class DataHandler:
    """Parameters and normalised (y, mT) histograms from Bulk results.

    ``histograms[pdg]`` has shape (n_runs, n_y, n_mt) and holds
    d²N/(dy dmT) per event.
    """

    def __init__(self, data, parameter_paths):
        self.parameter_paths = dict(parameter_paths)
        self.parameter_names = list(self.parameter_paths)

        self.y_edges = {}
        self.mt_edges = {}

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

        n_runs = len(parameters)
        missing = {pdg: n_runs - len(h) for pdg, h in histograms.items() if len(h) != n_runs}
        if missing:
            raise ValueError(f"PDGs missing in some runs (pdg: runs missing): {missing}")

        return (
            np.asarray(parameters, dtype=float),
            {pdg: np.stack(h) for pdg, h in histograms.items()},
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
        raise ValueError(f"unknown observable {name!r}")


def _pythia_setting(entries, key):
    """Value of ``key`` in a list of PYTHIA ``"key = value"`` strings."""
    for entry in entries:
        name, separator, value = entry.partition("=")
        if separator and name.strip() == key:
            return value.strip()
    raise KeyError(key)
