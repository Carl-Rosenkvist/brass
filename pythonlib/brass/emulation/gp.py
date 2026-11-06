"""Exact Gaussian-process regression for emulators.

Hyperparameters are optimised with scikit-learn; prediction, saving and
loading use plain NumPy/SciPy, so a fitted model is just a few arrays and
does not depend on pickled scikit-learn objects.
"""

import warnings

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

# Added to the diagonal of the training covariance for numerical stability
# (the same value scikit-learn uses by default).
JITTER = 1e-10


def matern52(A, B, length_scale):
    """Matérn ν=5/2 correlation between the rows of A and B (ARD)."""
    A = A / length_scale
    B = B / length_scale
    d2 = (
        np.sum(A * A, axis=1)[:, None]
        + np.sum(B * B, axis=1)[None, :]
        - 2.0 * (A @ B.T)
    )
    r = np.sqrt(5.0 * np.maximum(d2, 0.0))
    return (1.0 + r + r * r / 3.0) * np.exp(-r)


class GaussianProcess:
    """Single-output GP with kernel ``amplitude * Matern52 + noise``.

    The noise term absorbs statistical fluctuations of the training data
    (finite number of events), so the GP does not interpolate them. Predicted
    variances are those of the noise-free latent function.
    """

    def __init__(self, noise=True, n_restarts=0, random_state=None):
        self.noise = noise
        self.n_restarts = n_restarts
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel,
            Matern,
            WhiteKernel,
        )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        y_mean = float(np.mean(y))
        y_std = float(np.std(y)) or 1.0
        ys = (y - y_mean) / y_std

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(X.shape[1]),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        if self.noise:
            kernel = kernel + WhiteKernel(1e-2, (1e-8, 1e1))

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=JITTER,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
        )

        with warnings.catch_warnings():
            # A length scale at its upper bound only means the output does
            # not depend on that parameter.
            warnings.simplefilter("ignore", ConvergenceWarning)
            gpr.fit(X, ys)

        k = gpr.kernel_
        signal, noise = (k.k1, k.k2.noise_level) if self.noise else (k, 0.0)

        self._set_state(
            X=X,
            y=ys,
            y_mean=y_mean,
            y_std=y_std,
            amplitude=float(signal.k1.constant_value),
            length_scale=np.atleast_1d(signal.k2.length_scale).astype(float),
            noise_level=float(noise),
        )
        return self

    def _set_state(self, X, y, y_mean, y_std, amplitude, length_scale, noise_level):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.amplitude = float(amplitude)
        self.length_scale = np.asarray(length_scale, dtype=float)
        self.noise_level = float(noise_level)

        K = self.amplitude * matern52(self.X_train, self.X_train, self.length_scale)
        K[np.diag_indices_from(K)] += self.noise_level + JITTER

        self._L = cholesky(K, lower=True)
        self._alpha = cho_solve((self._L, True), self.y_train)

    def predict(self, X, return_var=False):
        """Mean (and variance) of the latent function at the rows of X."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Ks = self.amplitude * matern52(X, self.X_train, self.length_scale)

        mean = self.y_mean + self.y_std * (Ks @ self._alpha)
        if not return_var:
            return mean

        v = solve_triangular(self._L, Ks.T, lower=True, check_finite=False)
        var = np.maximum(self.amplitude - np.sum(v * v, axis=0), 0.0)
        return mean, var * self.y_std**2

    def state(self):
        return {
            "X": self.X_train,
            "y": self.y_train,
            "y_mean": np.float64(self.y_mean),
            "y_std": np.float64(self.y_std),
            "amplitude": np.float64(self.amplitude),
            "length_scale": self.length_scale,
            "noise_level": np.float64(self.noise_level),
        }

    @classmethod
    def from_state(cls, state):
        gp = cls(noise=float(state["noise_level"]) > 0.0)
        gp._set_state(
            X=state["X"],
            y=state["y"],
            y_mean=state["y_mean"],
            y_std=state["y_std"],
            amplitude=state["amplitude"],
            length_scale=state["length_scale"],
            noise_level=state["noise_level"],
        )
        return gp
