import numpy as np


class Likelihood:
    def __init__(
        self,
        rel_error_floor=0.05,
        abs_error_floor=0.05,
    ):
        self.rel_error_floor = rel_error_floor
        self.abs_error_floor = abs_error_floor

    def _safe_errors(self, y_exp, yerr_exp):
        return np.maximum.reduce(
            [
                np.abs(yerr_exp),
                self.rel_error_floor * np.abs(y_exp),
                np.full_like(
                    yerr_exp,
                    self.abs_error_floor,
                    dtype=float,
                ),
            ]
        )

    def contributions(
        self,
        x_model,
        y_model,
        x_exp,
        y_exp,
        yerr_exp,
        yerr_model=None,
    ):
        x_model = np.asarray(x_model)
        y_model = np.asarray(y_model)
        x_exp = np.asarray(x_exp)
        y_exp = np.asarray(y_exp)
        yerr_exp = np.asarray(yerr_exp)

        mask = (
            np.isfinite(x_exp)
            & np.isfinite(y_exp)
            & np.isfinite(yerr_exp)
            & (yerr_exp >= 0)
            & (x_exp >= x_model.min())
            & (x_exp <= x_model.max())
        )

        x_exp = x_exp[mask]
        y_exp = y_exp[mask]
        yerr_exp = yerr_exp[mask]

        y_model_at_exp = np.interp(
            x_exp,
            x_model,
            y_model,
        )

        error = self._safe_errors(
            y_exp,
            yerr_exp,
        )

        if yerr_model is not None:
            yerr_model_at_exp = np.interp(
                x_exp,
                x_model,
                np.asarray(yerr_model),
            )

            error = np.sqrt(error**2 + yerr_model_at_exp**2)

        residual = (y_model_at_exp - y_exp) / error

        return x_exp, residual**2

    def chi2(
        self,
        x_model,
        y_model,
        x_exp,
        y_exp,
        yerr_exp,
        yerr_model=None,
    ):
        _, chi2_i = self.contributions(
            x_model,
            y_model,
            x_exp,
            y_exp,
            yerr_exp,
            yerr_model,
        )

        return np.sum(chi2_i)

    def log_likelihood(
        self,
        x_model,
        y_model,
        x_exp,
        y_exp,
        yerr_exp,
        yerr_model=None,
    ):
        return -0.5 * self.chi2(
            x_model,
            y_model,
            x_exp,
            y_exp,
            yerr_exp,
            yerr_model,
        )


def _interpolation_matrix(x_new, x):
    """Matrix W with ``W @ f == np.interp(x_new, x, f)`` for any f."""
    W = np.zeros((len(x_new), len(x)))
    for j in range(len(x)):
        unit = np.zeros(len(x))
        unit[j] = 1.0
        W[:, j] = np.interp(x_new, x, unit)
    return W


class EmulatorLikelihood:
    """Gaussian log-likelihood of experimental data given an emulator.

    Datasets are added with :meth:`add`; calling the object with an array of
    parameter vectors (e.g. all MCMC walkers) returns their log-likelihoods::

        like = EmulatorLikelihood(emu, rel_error_floor=0.03)
        like.add(211, x=y_exp, y=dndy_exp, yerr=err_exp, fixed={"Sqrtsnn": 17.3})
        logp = like(thetas)          # thetas: (n, len(like.parameters))

    For each dataset the model is interpolated (linearly) to the experimental
    points, and

        log L = -1/2 [ r^T C^-1 r + log det C + n log 2π ],
        C = diag(experimental errors²) + emulator covariance,

    with residuals r = model - data. The emulator covariance includes the
    GP uncertainty and the PCA truncation error and is correlated between
    points; ``covariance="diagonal"`` keeps only its diagonal.

    Parameters
    ----------
    emulator
        A fitted emulator (e.g. :class:`~brass.emulation.bulk.BulkEmulator`).
    parameters
        Names of the parameters in ``thetas``, in order. Default: all
        emulator parameters that are not fixed by any dataset.
    rel_error_floor, abs_error_floor
        Lower bounds on experimental errors: ``max(yerr, rel*|y|, abs)``.
    covariance
        ``"full"``, ``"diagonal"`` or ``"none"`` (ignore emulator uncertainty).
    """

    def __init__(
        self,
        emulator,
        parameters=None,
        rel_error_floor=0.0,
        abs_error_floor=0.0,
        covariance="full",
    ):
        if covariance not in ("full", "diagonal", "none"):
            raise ValueError("covariance must be 'full', 'diagonal' or 'none'")

        self.emulator = emulator
        self.rel_error_floor = rel_error_floor
        self.abs_error_floor = abs_error_floor
        self.covariance = covariance
        self._parameters = None if parameters is None else list(parameters)
        self.datasets = []

    @property
    def parameters(self):
        if self._parameters is not None:
            return self._parameters
        fixed = {name for d in self.datasets for name in d["fixed"]}
        return [p for p in self.emulator.parameter_names if p not in fixed]

    def add(self, pdg, x, y, yerr, observable="dNdy", fixed=None, label=None):
        """Add one experimental dataset.

        ``fixed`` gives values of emulator parameters that are not fitted,
        such as the collision energy of this dataset. Points outside the
        emulated range, or with non-finite values, are dropped.
        """
        fixed = dict(fixed or {})
        unknown = set(fixed) - set(self.emulator.parameter_names)
        if unknown:
            raise KeyError(f"unknown fixed parameters: {sorted(unknown)}")

        model, A = self.emulator.linear_model(pdg, observable)

        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        yerr = np.atleast_1d(np.asarray(yerr, dtype=float))

        axis = self.emulator.axis(pdg, observable)
        mask = np.isfinite(y) & np.isfinite(yerr) & (yerr >= 0)
        if axis is None:
            W = np.ones((len(y), 1))
        else:
            mask &= np.isfinite(x) & (x >= axis.min()) & (x <= axis.max())
            W = _interpolation_matrix(x[mask], axis)
        if axis is None:
            W = W[mask]

        A_total = W if A is None else W @ A
        error = np.maximum.reduce(
            [
                np.abs(yerr[mask]),
                self.rel_error_floor * np.abs(y[mask]),
                np.full(mask.sum(), float(self.abs_error_floor)),
            ]
        )

        self.datasets.append(
            {
                "label": label or f"{pdg} {observable} {fixed}",
                "pdg": pdg,
                "observable": observable,
                "fixed": fixed,
                "model": model,
                "map": model.linear_map(A_total),
                "x": x[mask],
                "y": y[mask],
                "variance": error**2,
            }
        )
        return self

    @property
    def n_points(self):
        return sum(len(d["y"]) for d in self.datasets)

    def _inputs(self, thetas, fixed):
        names = self.emulator.parameter_names
        free = self.parameters
        missing = set(names) - set(free) - set(fixed)
        if missing:
            raise ValueError(f"parameters neither fitted nor fixed: {sorted(missing)}")

        X = np.empty((len(thetas), len(names)))
        for i, name in enumerate(names):
            X[:, i] = fixed[name] if name in fixed else thetas[:, free.index(name)]
        return X

    def _latent(self, thetas):
        """GP predictions for every (model, fixed values), one call per model."""
        requests = {}
        for d in self.datasets:
            key = tuple(sorted(d["fixed"].items()))
            requests.setdefault(id(d["model"]), (d["model"], {}))[1][key] = d["fixed"]

        latent = {}
        n = len(thetas)
        for model_id, (model, fixed_sets) in requests.items():
            keys = list(fixed_sets)
            X = np.vstack([self._inputs(thetas, fixed_sets[k]) for k in keys])
            z_mean, z_var = model.predict_latent(X)
            for i, key in enumerate(keys):
                latent[(model_id, key)] = (
                    z_mean[i * n : (i + 1) * n],
                    z_var[i * n : (i + 1) * n],
                )
        return latent

    def evaluate(self, thetas):
        """Per-dataset χ² and log-likelihood, each of shape (n_datasets, n)."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        if thetas.shape[1] != len(self.parameters):
            raise ValueError(
                f"thetas must have {len(self.parameters)} columns: {self.parameters}"
            )

        latent = self._latent(thetas)
        chi2 = np.zeros((len(self.datasets), len(thetas)))
        logl = np.zeros_like(chi2)

        for i, d in enumerate(self.datasets):
            n_points = len(d["y"])
            if n_points == 0:
                continue

            z_mean, z_var = latent[(id(d["model"]), tuple(sorted(d["fixed"].items())))]
            r = None
            if self.covariance == "full":
                mean, cov = d["map"](z_mean, z_var, return_cov=True)
                cov[:, np.arange(n_points), np.arange(n_points)] += d["variance"]
                r = mean - d["y"]
                sign, logdet = np.linalg.slogdet(cov)
                solved = np.linalg.solve(cov, r[..., None])[..., 0]
                chi2[i] = np.einsum("ni,ni->n", r, solved)
                logdet = np.where(sign > 0, logdet, np.nan)
            else:
                if self.covariance == "diagonal":
                    mean, std = d["map"](z_mean, z_var, return_std=True)
                    var = d["variance"] + std**2
                else:
                    mean = d["map"](z_mean)
                    var = np.broadcast_to(d["variance"], mean.shape)
                r = mean - d["y"]
                chi2[i] = np.sum(r * r / var, axis=1)
                logdet = np.sum(np.log(var), axis=1)

            logl[i] = -0.5 * (chi2[i] + logdet + n_points * np.log(2 * np.pi))

        logl[~np.isfinite(logl)] = -np.inf
        return chi2, logl

    def chi2(self, thetas):
        return self.evaluate(thetas)[0].sum(axis=0)

    def __call__(self, thetas):
        """Total log-likelihood for each row of thetas."""
        return self.evaluate(thetas)[1].sum(axis=0)
