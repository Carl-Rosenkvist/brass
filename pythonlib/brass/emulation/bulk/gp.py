import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler


class GPEmulator:
    def __init__(self):

        self.scaler = StandardScaler()
        self.models = {}

    def fit(self, parameters, latent):

        X = self.scaler.fit_transform(parameters)

        for pdg, Z in latent.items():
            self.models[pdg] = []

            for i in range(Z.shape[1]):
                kernel = Matern(
                    length_scale=np.ones(X.shape[1]),
                    nu=2.5,
                )

                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                )

                gp.fit(
                    X,
                    Z[:, i],
                )

                self.models[pdg].append(gp)

        return self

    def predict(
        self,
        parameters,
        pdg,
        return_std=False,
    ):

        X = np.atleast_2d(parameters)
        X = self.scaler.transform(X)

        if not return_std:
            predictions = [gp.predict(X) for gp in self.models[pdg]]

            return np.column_stack(predictions)

        # Mean + GP uncertainty
        means = []
        stds = []

        for gp in self.models[pdg]:
            mean, std = gp.predict(
                X,
                return_std=True,
            )

            means.append(mean)
            stds.append(std)

        mean = np.column_stack(means)
        std = np.column_stack(stds)

        return mean, std

    def sample(
        self,
        parameters,
        pdg,
        n_samples=1000,
        random_state=None,
    ):

        mean, std = self.predict(
            parameters,
            pdg,
            return_std=True,
        )

        rng = np.random.default_rng(random_state)

        samples = rng.normal(
            loc=mean[:, None, :],
            scale=std[:, None, :],
            size=(
                mean.shape[0],
                n_samples,
                mean.shape[1],
            ),
        )

        return samples
