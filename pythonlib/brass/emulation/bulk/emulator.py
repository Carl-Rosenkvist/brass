import pickle

import numpy as np

from .data import DataHandler
from .gp import GPEmulator
from .observables import Observables


class BulkEmulator:
    def __init__(
        self,
        data,
        parameter_paths,
        n_components=3,
    ):
        self.data = DataHandler(
            data=data,
            parameter_paths=parameter_paths,
            n_components=n_components,
        )

        self.gp = GPEmulator()

    @classmethod
    def from_file(
        cls,
        path,
        parameter_paths,
        n_components=3,
    ):
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            data=data,
            parameter_paths=parameter_paths,
            n_components=n_components,
        )

    def fit(self):
        self.gp.fit(
            self.data.parameters,
            self.data.latent,
        )

        return self

    def predict(self, parameters, pdg):
        parameters = np.atleast_2d(parameters)

        z = self.gp.predict(
            parameters,
            pdg,
        )

        return self.data.inverse_transform(
            pdg,
            z,
        )

    def observables(self, pdg):
        return Observables(
            self.data.y_edges[pdg],
            self.data.mt_edges[pdg],
        )

    def dNdy(self, parameters, pdg):
        histograms = self.predict(
            parameters,
            pdg,
        )

        obs = self.observables(pdg)

        values = np.array([obs.dNdy(hist) for hist in histograms])

        return obs.y, values

    def dNdy_with_std(self, parameters, pdg):
        parameters = np.atleast_2d(parameters)

        z_mean, z_std = self.gp.predict(
            parameters,
            pdg,
            return_std=True,
        )

        pca = self.data.pca[pdg]
        obs = self.observables(pdg)

        # Mean histogram -> mean dN/dy
        histograms = self.data.inverse_transform(
            pdg,
            z_mean,
        )

        mean = np.array([obs.dNdy(hist) for hist in histograms])

        # Reshape PCA components into histogram shape
        components = pca.components_.reshape(
            pca.n_components_,
            len(obs.y),
            len(obs.mt),
        )

        # Effect of each PCA component on each dN/dy bin
        components_dNdy = np.array([obs.dNdy(component) for component in components])
        # shape: (n_components, n_y)

        # Var[dN/dy_j] = sum_k sigma_z_k^2 * component_kj^2
        var = (z_std**2) @ (components_dNdy**2)

        std = np.sqrt(var)

        return obs.y, mean, std

    def dNdy_with_covariance(
        self,
        parameters,
        pdg,
    ):
        parameters = np.atleast_2d(parameters)

        z_mean, z_std = self.gp.predict(
            parameters,
            pdg,
            return_std=True,
        )

        pca = self.data.pca[pdg]
        obs = self.observables(pdg)

        histograms = self.data.inverse_transform(
            pdg,
            z_mean,
        )

        n_bins = histograms[0].size
        n_y = len(obs.y)

        A = np.zeros((n_y, n_bins))

        for i in range(n_bins):
            basis = np.zeros_like(histograms[0])
            basis.flat[i] = 1.0
            A[:, i] = obs.dNdy(basis)

        means = []
        covariances = []

        for hist, std in zip(
            histograms,
            z_std,
        ):
            cov_z = np.diag(std**2)

            cov_hist = pca.components_.T @ cov_z @ pca.components_

            means.append(obs.dNdy(hist))

            covariances.append(A @ cov_hist @ A.T)

        return (
            obs.y,
            np.stack(means),
            np.stack(covariances),
        )

    def dNdmt(self, parameters, pdg):
        histograms = self.predict(
            parameters,
            pdg,
        )

        obs = self.observables(pdg)

        values = np.array([obs.dNdmt(hist) for hist in histograms])

        return obs.mt, values

    def yield_total(self, parameters, pdg):
        histograms = self.predict(
            parameters,
            pdg,
        )

        obs = self.observables(pdg)

        return np.array([obs.yield_total(hist) for hist in histograms])

    def validate_covariance(
        self,
        parameters,
        pdg,
        n_samples=100000,
        random_state=42,
    ):
        z_mean, z_std = self.gp.predict(
            parameters,
            pdg,
            return_std=True,
        )

        z_samples = self.gp.sample(
            parameters,
            pdg,
            n_samples=n_samples,
            random_state=random_state,
        )

        pca = self.data.pca[pdg]
        results = []

        for mean, std, samples in zip(
            z_mean,
            z_std,
            z_samples,
        ):
            hist_mean = self.data.inverse_transform(
                pdg,
                mean,
            )[0]

            hist_samples = self.data.inverse_transform(
                pdg,
                samples,
            )

            hist_samples_flat = hist_samples.reshape(
                n_samples,
                -1,
            )

            cov_mc = np.cov(
                hist_samples_flat,
                rowvar=False,
            )

            cov_z = np.diag(std**2)

            cov_analytic = pca.components_.T @ cov_z @ pca.components_

            mean_mc = hist_samples.mean(axis=0)

            covariance_error = np.linalg.norm(cov_mc - cov_analytic) / np.linalg.norm(
                cov_analytic
            )

            mean_error = np.max(np.abs(mean_mc - hist_mean))

            results.append(
                {
                    "mean": hist_mean,
                    "cov_analytic": cov_analytic,
                    "cov_mc": cov_mc,
                    "mean_error": mean_error,
                    "covariance_error": covariance_error,
                }
            )

        return results
