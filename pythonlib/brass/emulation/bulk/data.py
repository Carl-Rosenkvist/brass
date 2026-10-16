import pickle

import numpy as np
from sklearn.decomposition import PCA


class DataHandler:
    def __init__(
        self,
        data,
        parameter_paths,
        n_components=3,
    ):
        self.parameter_paths = parameter_paths
        self.parameter_names = list(parameter_paths)
        self.n_components = n_components

        self.y_edges = {}
        self.mt_edges = {}

        self.parameters, self.histograms = self._process(data)

        self.pca = {}
        self.latent = {}

        self._fit_pca()

    @classmethod
    def from_file(
        cls,
        path,
        parameter_paths,
        n_components=3,
    ):
        with open(path, "rb") as f:
            data = pickle.load(f)

        return (
            cls(
                data=data,
                parameter_paths=parameter_paths,
                n_components=n_components,
            )
            @ staticmethod
        )

    def _get_nested(self, mapping, path):
        for key in path:
            if isinstance(mapping, list) and isinstance(key, str):
                # Extract a named PYTHIA setting from a list of strings.
                settings = {}

                for entry in mapping:
                    name, separator, value = entry.partition("=")

                    if separator:
                        settings[name.strip()] = value.strip()

                mapping = float(settings[key])

            else:
                mapping = mapping[key]

        return mapping

    def _extract_parameters(self, analysis):

        return np.array(
            [
                self._get_nested(
                    analysis.config,
                    self.parameter_paths[name],
                )
                for name in self.parameter_names
            ]
        )

    def _process(self, data):

        parameters = []
        histograms = {}

        for result in data:
            analysis = result["result"]

            parameters.append(self._extract_parameters(analysis))

            for pdg, histo in analysis.histograms.items():
                counts = np.asarray(
                    histo.values,
                    dtype=float,
                )

                y_edges, mt_edges = map(
                    np.asarray,
                    histo.edges,
                )

                dy = np.diff(y_edges)
                dmt = np.diff(mt_edges)

                hist = counts / (analysis.events * dy[:, None] * dmt[None, :])

                histograms.setdefault(pdg, []).append(hist)

                if pdg not in self.y_edges:
                    self.y_edges[pdg] = y_edges
                    self.mt_edges[pdg] = mt_edges

        parameters = np.stack(parameters)

        histograms = {pdg: np.stack(hists) for pdg, hists in histograms.items()}

        return parameters, histograms

    def _fit_pca(self):

        for pdg, H in self.histograms.items():
            H_flat = H.reshape(
                len(H),
                -1,
            )

            pca = PCA(n_components=self.n_components)

            Z = pca.fit_transform(H_flat)

            self.pca[pdg] = pca
            self.latent[pdg] = Z

    def inverse_transform(self, pdg, Z):

        Z = np.atleast_2d(Z)

        H_flat = self.pca[pdg].inverse_transform(Z)

        hist_shape = self.histograms[pdg].shape[1:]

        return H_flat.reshape(
            -1,
            *hist_shape,
        )
