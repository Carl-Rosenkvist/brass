"""Exact Gaussian-process regression for emulators.

The kernel is ``amplitude * Matern52 + noise`` with one length scale per
input. Its hyperparameters maximise the log marginal likelihood (L-BFGS-B
with the analytic gradient, Rasmussen & Williams 2006, eq. 5.9), within the
bounds of :data:`BOUNDS`. A fitted model is just a few arrays.
"""

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.linalg.lapack import dpotri
from scipy.optimize import minimize

# Added to the diagonal of the training covariance for numerical stability.
JITTER = 1e-10

# Search ranges of the hyperparameters (outputs are standardised and inputs
# scaled to unit standard deviation before fitting).
BOUNDS = {
    "amplitude": (1e-3, 1e3),
    "length_scale": (1e-2, 1e3),
    "noise_level": (1e-8, 1e1),
}


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


def _negative_log_likelihood(log_p, X, y, noise):
    """-log p(y | X, hyperparameters) and its gradient in log(hyperparameters).

    ``log_p`` = log(amplitude, length scales..., noise level if ``noise``).
    """
    n, d = X.shape
    amplitude = np.exp(log_p[0])
    length_scale = np.exp(log_p[1 : 1 + d])
    noise_level = np.exp(log_p[-1]) if noise else 0.0

    A = X / length_scale
    sq = np.sum(A * A, axis=1)
    r = np.sqrt(5.0 * np.maximum(sq[:, None] + sq[None, :] - 2.0 * (A @ A.T), 0.0))
    e = np.exp(-r)
    K = amplitude * (1.0 + r + r * r / 3.0) * e  # signal covariance
    C = K.copy()
    C[np.diag_indices(n)] += noise_level + JITTER
    try:
        L = cholesky(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf, np.zeros_like(log_p)

    alpha = cho_solve((L, True), y, check_finite=False)
    value = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2.0 * np.pi)

    # d(-log L)/dC = (C^-1 - alpha alpha^T) / 2 =: W / 2
    W, info = dpotri(L, lower=1)
    if info != 0:
        return np.inf, np.zeros_like(log_p)
    W = np.tril(W) + np.tril(W, -1).T - np.outer(alpha, alpha)

    grad = np.empty_like(log_p)
    grad[0] = 0.5 * np.sum(W * K)  # dC/dlog(amplitude) = K
    # dC/dlog(l_k) = amplitude * 5/3 (1 + r) exp(-r) * (x_ik - x_jk)^2 / l_k^2
    F = W * (amplitude * (5.0 / 3.0) * (1.0 + r) * e)
    for k in range(d):
        diff = A[:, k][:, None] - A[:, k][None, :]
        grad[1 + k] = 0.5 * np.sum(F * diff * diff)
    if noise:
        grad[-1] = 0.5 * noise_level * np.trace(W)
    return value, grad


def optimise(X, y, noise=True, n_restarts=0, random_state=None):
    """Hyperparameters that maximise the log marginal likelihood of (X, y).

    The search starts from amplitude 1, length scales 1 and noise 0.01 (the
    scikit-learn defaults), plus ``n_restarts`` random starts uniform in
    log(bounds). Returns ``{"amplitude", "length_scale", "noise_level"}``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    bounds = [BOUNDS["amplitude"]] + [BOUNDS["length_scale"]] * d
    start = [1.0] + [1.0] * d
    if noise:
        bounds.append(BOUNDS["noise_level"])
        start.append(1e-2)
    bounds = np.log(np.asarray(bounds))

    rng = np.random.default_rng(random_state)
    starts = [np.log(start)] + [
        rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(n_restarts)
    ]
    best = None
    for x0 in starts:
        result = minimize(
            _negative_log_likelihood, x0, args=(X, y, noise), jac=True,
            method="L-BFGS-B", bounds=bounds,
        )  # fmt: skip
        if best is None or result.fun < best.fun:
            best = result
    p = np.exp(best.x)
    return {
        "amplitude": float(p[0]),
        "length_scale": p[1 : 1 + d],
        "noise_level": float(p[-1]) if noise else 0.0,
    }


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

    @staticmethod
    def standardise(y):
        """(standardised y, mean, std) as used for fitting."""
        y = np.asarray(y, dtype=float)
        y_mean = float(np.mean(y))
        y_std = float(np.std(y)) or 1.0
        return (y - y_mean) / y_std, y_mean, y_std

    def fit(self, X, y):
        ys, y_mean, y_std = self.standardise(y)
        hyper = optimise(X, ys, self.noise, self.n_restarts, self.random_state)
        self._set_state(X=X, y=ys, y_mean=y_mean, y_std=y_std, **hyper)
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


def fit_gps(tasks, noise=True, n_restarts=0, random_state=None, n_jobs=1):
    """Fit one GP per ``(X, y)`` in ``tasks``, in parallel processes.

    ``n_jobs`` as in joblib (-1: all cores). The processes only optimise the
    hyperparameters; the GPs are assembled here, so large matrices are not
    sent between processes. Returns the GPs in the order of ``tasks``.
    """
    standardised = [(X, *GaussianProcess.standardise(y)) for X, y in tasks]
    args = (noise, n_restarts, random_state)
    if n_jobs == 1 or len(tasks) <= 1:
        hypers = [optimise(X, ys, *args) for X, ys, _, _ in standardised]
    else:
        from joblib import Parallel, delayed

        hypers = Parallel(n_jobs=n_jobs)(
            delayed(optimise)(X, ys, *args) for X, ys, _, _ in standardised
        )
    gps = []
    for (X, ys, y_mean, y_std), hyper in zip(standardised, hypers):
        gp = GaussianProcess(noise, n_restarts, random_state)
        gp._set_state(X=X, y=ys, y_mean=y_mean, y_std=y_std, **hyper)
        gps.append(gp)
    return gps
