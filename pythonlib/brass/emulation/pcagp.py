"""PCA + Gaussian-process emulator for vector-valued model output."""

import numpy as np

from .gp import GaussianProcess


class LinearMap:
    """Emulator output mapped through a fixed matrix A: ``A @ output``.

    Precomputes everything that does not depend on the parameters, so that
    mapping latent predictions costs only small matrix products.
    """

    def __init__(self, model, A=None):
        components = model.components  # (k, m)
        truncation = model.truncation_cov  # (m, m)

        if A is None:
            self.mean = model.output_mean
            self.B = components
            self.truncation = truncation
        else:
            A = np.atleast_2d(np.asarray(A, dtype=float))
            self.mean = A @ model.output_mean
            self.B = components @ A.T
            self.truncation = A @ truncation @ A.T

    def __call__(self, z_mean, z_var=None, *, return_std=False, return_cov=False):
        mean = self.mean + z_mean @ self.B
        if return_cov:
            cov = np.einsum("nk,ki,kj->nij", z_var, self.B, self.B)
            return mean, cov + self.truncation
        if return_std:
            var = z_var @ (self.B * self.B) + np.diag(self.truncation)
            return mean, np.sqrt(var)
        return mean


class PCAGP:
    """Emulates ``Y = f(X)`` for vector output Y.

    The training outputs are centred and decomposed into principal
    components; one GP is fitted per kept component. The predicted
    uncertainty combines the GP variances of the kept components with the
    covariance of the discarded ones (truncation error).

    Parameters
    ----------
    n_components
        Number of components (int), or the fraction of variance to keep
        (float in (0, 1)).
    log_parameters
        Indices of input columns that are emulated in ``log(x)``.
    noise
        Fit a noise term with each GP (recommended for Monte Carlo output).
    include_truncation
        Add the covariance of discarded components to predictions.
    """

    def __init__(
        self,
        n_components=0.99,
        log_parameters=(),
        noise=True,
        n_restarts=0,
        random_state=None,
        include_truncation=True,
    ):
        self.n_components = n_components
        self.log_parameters = tuple(int(i) for i in log_parameters)
        self.noise = noise
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.include_truncation = include_truncation

    # -- inputs ---------------------------------------------------------------

    def _raw_inputs(self, X):
        X = np.array(np.atleast_2d(X), dtype=float)
        if self.log_parameters:
            cols = list(self.log_parameters)
            if np.any(X[:, cols] <= 0):
                raise ValueError("log-scaled parameters must be positive")
            X[:, cols] = np.log(X[:, cols])
        return X

    def _inputs(self, X):
        return (self._raw_inputs(X) - self.x_mean) / self.x_scale

    # -- fitting --------------------------------------------------------------

    def _n_kept(self, explained):
        total = explained.sum()
        if total <= 0:
            return 0
        n = self.n_components
        if isinstance(n, float) and 0 < n < 1:
            fraction = np.cumsum(explained) / total
            n = int(np.searchsorted(fraction, n) + 1)
        return int(min(n, len(explained)))

    def fit(self, X, Y):
        X = self._raw_inputs(X)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or len(Y) != len(X):
            raise ValueError("Y must have shape (n_runs, n_outputs) matching X")

        self.x_mean = X.mean(axis=0)
        self.x_scale = X.std(axis=0)
        self.x_scale[self.x_scale == 0] = 1.0
        Xs = (X - self.x_mean) / self.x_scale

        self.output_mean = Y.mean(axis=0)
        Yc = Y - self.output_mean

        _, S, Vt = np.linalg.svd(Yc, full_matrices=False)
        explained = S**2 / max(len(Y) - 1, 1)
        k = self._n_kept(explained)

        self.components = Vt[:k]
        self.explained_variance = explained[:k]
        self.explained_fraction = (
            explained[:k].sum() / explained.sum() if explained.sum() > 0 else 1.0
        )

        Z = Yc @ self.components.T
        residual = Yc - Z @ self.components
        m = Y.shape[1]
        self.truncation_cov = (
            residual.T @ residual / max(len(Y) - 1, 1)
            if self.include_truncation
            else np.zeros((m, m))
        )

        self.gps = [
            GaussianProcess(
                noise=self.noise,
                n_restarts=self.n_restarts,
                random_state=self.random_state,
            ).fit(Xs, Z[:, i])
            for i in range(k)
        ]
        return self

    # -- prediction -----------------------------------------------------------

    @property
    def n_kept(self):
        return len(self.gps)

    def predict_latent(self, X):
        """Means and variances of the PCA components, each (n, n_kept)."""
        Xs = self._inputs(X)
        n = len(Xs)
        z_mean = np.zeros((n, self.n_kept))
        z_var = np.zeros((n, self.n_kept))
        for i, gp in enumerate(self.gps):
            z_mean[:, i], z_var[:, i] = gp.predict(Xs, return_var=True)
        return z_mean, z_var

    def linear_map(self, A=None):
        return LinearMap(self, A)

    def predict(self, X, A=None, *, return_std=False, return_cov=False):
        """Predicted output (optionally mapped through A) for the rows of X.

        Returns the mean, plus the standard deviation or the full covariance
        if requested.
        """
        z_mean, z_var = self.predict_latent(X)
        return self.linear_map(A)(
            z_mean, z_var, return_std=return_std, return_cov=return_cov
        )

    # -- persistence ----------------------------------------------------------

    _ARRAYS = (
        "x_mean",
        "x_scale",
        "output_mean",
        "components",
        "explained_variance",
        "truncation_cov",
    )

    def state(self):
        out = {name: getattr(self, name) for name in self._ARRAYS}
        out["log_parameters"] = np.array(self.log_parameters, dtype=int)
        out["explained_fraction"] = np.float64(self.explained_fraction)
        out["include_truncation"] = np.bool_(self.include_truncation)
        for i, gp in enumerate(self.gps):
            for key, value in gp.state().items():
                out[f"gp{i}/{key}"] = value
        return out

    @classmethod
    def from_state(cls, state):
        model = cls(
            n_components=len(state["components"]),
            log_parameters=state["log_parameters"],
            include_truncation=bool(state["include_truncation"]),
        )
        for name in cls._ARRAYS:
            setattr(model, name, np.asarray(state[name]))
        model.explained_fraction = float(state["explained_fraction"])
        model.gps = []
        for i in range(len(model.components)):
            prefix = f"gp{i}/"
            model.gps.append(
                GaussianProcess.from_state(
                    {
                        key[len(prefix) :]: value
                        for key, value in state.items()
                        if key.startswith(prefix)
                    }
                )
            )
        return model
