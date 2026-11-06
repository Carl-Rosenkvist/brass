import json

import numpy as np

from ..pcagp import PCAGP
from .data import DataHandler
from .observables import Observables

OBSERVABLES = ("dNdy", "dNdmt", "hist2d")


class BulkEmulator:
    """Emulates Bulk-analysis spectra as a function of model parameters.

    One :class:`~brass.emulation.pcagp.PCAGP` is fitted per particle species
    and observable. Supported observables:

    ``"dNdy"``
        rapidity spectrum (summed over mT) — the cheapest and least noisy
        choice when only dN/dy is compared to data;
    ``"dNdmt"``
        transverse-mass spectrum (summed over y);
    ``"hist2d"``
        the full d²N/(dy dmT) histogram. dN/dy, dN/dmT and yields can be
        derived from it, but rare particles need many PCA components.

    Every parameter in ``parameter_paths`` is a GP input, including e.g. the
    collision energy, so one emulator covers an energy scan.
    """

    def __init__(
        self,
        data,
        parameter_paths,
        observables=("dNdy",),
        n_components=0.99,
        log_parameters=(),
        noise=True,
        n_restarts=0,
        random_state=None,
        include_truncation=True,
    ):
        self.data = (
            data
            if isinstance(data, DataHandler)
            else DataHandler(data=data, parameter_paths=parameter_paths)
        )

        observables = (observables,) if isinstance(observables, str) else observables
        unknown = set(observables) - set(OBSERVABLES)
        if unknown:
            raise ValueError(
                f"unknown observables {sorted(unknown)}; choose from {OBSERVABLES}"
            )

        self.parameter_names = list(self.data.parameter_names)
        unknown = set(log_parameters) - set(self.parameter_names)
        if unknown:
            raise ValueError(f"log_parameters not in parameter_paths: {sorted(unknown)}")

        self.observable_names = list(observables)
        self.log_parameters = list(log_parameters)
        self.y_edges = dict(self.data.y_edges)
        self.mt_edges = dict(self.data.mt_edges)
        self.settings = {
            "n_components": n_components,
            "noise": noise,
            "n_restarts": n_restarts,
            "random_state": random_state,
            "include_truncation": include_truncation,
        }
        self.models = {}

    @classmethod
    def from_file(cls, path, parameter_paths, **kwargs):
        return cls(DataHandler.from_file(path, parameter_paths), parameter_paths, **kwargs)

    # -- fitting --------------------------------------------------------------

    @property
    def pdgs(self):
        return list(self.y_edges)

    def _new_model(self):
        return PCAGP(
            log_parameters=[self.parameter_names.index(p) for p in self.log_parameters],
            **self.settings,
        )

    def fit(self, pdgs=None, verbose=False):
        """Fit one PCA+GP model per particle and observable."""
        if self.data is None:
            raise RuntimeError("a loaded emulator has no training data to fit")

        for pdg in self.pdgs if pdgs is None else pdgs:
            for name in self.observable_names:
                model = self._new_model().fit(
                    self.data.parameters, self.data.observable(pdg, name)
                )
                self.models[(pdg, name)] = model
                if verbose:
                    print(
                        f"{pdg:>7} {name:<7} {model.n_kept:3d} components "
                        f"({100 * model.explained_fraction:.1f} % of variance)"
                    )
        return self

    # -- observables ----------------------------------------------------------

    def observables(self, pdg):
        return Observables(self.y_edges[pdg], self.mt_edges[pdg])

    def axis(self, pdg, observable):
        """Bin centres of an observable (None for the total yield)."""
        obs = self.observables(pdg)
        return {"dNdy": obs.y, "dNdmt": obs.mt, "yield": None}.get(observable)

    def linear_model(self, pdg, observable):
        """Fitted model and matrix A with ``observable = A @ model output``.

        A is None when the observable is emulated directly. Otherwise it is
        derived linearly from another fitted observable (e.g. dN/dy from the
        2D histogram, or the yield from dN/dy).
        """
        if (pdg, observable) in self.models:
            return self.models[(pdg, observable)], None

        obs = self.observables(pdg)
        ny, nmt = len(obs.dy), len(obs.dmt)
        derived = {
            "dNdy": {"hist2d": np.kron(np.eye(ny), obs.dmt[None, :])},
            "dNdmt": {"hist2d": np.kron(obs.dy[None, :], np.eye(nmt))},
            "yield": {
                "dNdy": obs.dy[None, :],
                "dNdmt": obs.dmt[None, :],
                "hist2d": np.kron(obs.dy, obs.dmt)[None, :],
            },
        }
        for source, A in derived.get(observable, {}).items():
            if (pdg, source) in self.models:
                return self.models[(pdg, source)], A

        if pdg not in self.y_edges:
            raise KeyError(f"no emulator for PDG {pdg}")
        if not self.models:
            raise RuntimeError("emulator is not fitted; call fit() first")
        raise ValueError(
            f"{observable!r} for PDG {pdg} cannot be obtained from the fitted "
            f"observables {self.observable_names}"
        )

    def predict_observable(
        self, parameters, pdg, observable, *, return_std=False, return_cov=False
    ):
        """Mean (and std or covariance) of an observable, shape (n, n_bins)."""
        model, A = self.linear_model(pdg, observable)
        return model.predict(
            np.atleast_2d(parameters), A, return_std=return_std, return_cov=return_cov
        )

    # -- convenience ----------------------------------------------------------

    def predict(self, parameters, pdg):
        """Predicted 2D histograms d²N/(dy dmT), shape (n, n_y, n_mt)."""
        obs = self.observables(pdg)
        mean = self.predict_observable(parameters, pdg, "hist2d")
        return mean.reshape(-1, len(obs.y), len(obs.mt))

    def dNdy(self, parameters, pdg):
        return self.axis(pdg, "dNdy"), self.predict_observable(parameters, pdg, "dNdy")

    def dNdy_with_std(self, parameters, pdg):
        mean, std = self.predict_observable(parameters, pdg, "dNdy", return_std=True)
        return self.axis(pdg, "dNdy"), mean, std

    def dNdy_with_covariance(self, parameters, pdg):
        mean, cov = self.predict_observable(parameters, pdg, "dNdy", return_cov=True)
        return self.axis(pdg, "dNdy"), mean, cov

    def dNdmt(self, parameters, pdg):
        return self.axis(pdg, "dNdmt"), self.predict_observable(parameters, pdg, "dNdmt")

    def yield_total(self, parameters, pdg):
        return self.predict_observable(parameters, pdg, "yield")[:, 0]

    # -- validation -----------------------------------------------------------

    def validate(self, holdout=0.2, random_state=0, observables=None, pdgs=None):
        """Refit on a random subset of runs and predict the held-out runs.

        Returns ``{(pdg, observable): result}`` where result holds the
        held-out ``parameters``, simulated ``truth``, emulated ``mean`` and
        ``std``, and the standardised residuals ``(mean - truth) / std``. For
        a well-calibrated emulator these are roughly standard normal.

        All runs at a parameter point (e.g. the same point at several
        energies) are held out together.
        """
        if self.data is None:
            raise RuntimeError("a loaded emulator has no training data to validate")

        X = self.data.parameters

        # Parameters with few distinct values (e.g. a set of energies) form a
        # grid; runs that differ only in those belong to the same point.
        grid = [
            i for i in range(X.shape[1]) if len(np.unique(X[:, i])) <= 0.1 * len(X)
        ]
        keys = np.delete(X, grid, axis=1) if len(grid) < X.shape[1] else X
        _, point_of_run = np.unique(keys, axis=0, return_inverse=True)
        point_of_run = point_of_run.ravel()
        n_points = point_of_run.max() + 1

        rng = np.random.default_rng(random_state)
        n_test = min(max(1, int(round(holdout * n_points))), n_points - 1)
        test = np.isin(point_of_run, rng.choice(n_points, size=n_test, replace=False))

        results = {}
        for pdg in self.pdgs if pdgs is None else pdgs:
            for name in observables or self.observable_names:
                Y = self.data.observable(pdg, name)
                model = self._new_model().fit(X[~test], Y[~test])
                mean, std = model.predict(X[test], return_std=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = (mean - Y[test]) / std
                results[(pdg, name)] = {
                    "parameters": X[test],
                    "truth": Y[test],
                    "mean": mean,
                    "std": std,
                    "residuals": z,
                }
        return results

    # -- persistence ----------------------------------------------------------

    def save(self, path):
        """Save the fitted emulator (without training data) to a .npz file."""
        meta = {
            "format": 1,
            "parameter_names": self.parameter_names,
            "observables": self.observable_names,
            "log_parameters": self.log_parameters,
            "settings": self.settings,
            "models": [[int(pdg), name] for pdg, name in self.models],
            "pdgs": [int(pdg) for pdg in self.pdgs],
        }
        arrays = {"meta": np.array(json.dumps(meta))}
        for pdg in self.pdgs:
            arrays[f"edges/{pdg}/y"] = self.y_edges[pdg]
            arrays[f"edges/{pdg}/mt"] = self.mt_edges[pdg]
        for (pdg, name), model in self.models.items():
            for key, value in model.state().items():
                arrays[f"model/{pdg}/{name}/{key}"] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}

        meta = json.loads(str(arrays["meta"]))
        emu = cls.__new__(cls)
        emu.data = None
        emu.parameter_names = meta["parameter_names"]
        emu.observable_names = meta["observables"]
        emu.log_parameters = meta["log_parameters"]
        emu.settings = meta["settings"]
        emu.y_edges = {pdg: arrays[f"edges/{pdg}/y"] for pdg in meta["pdgs"]}
        emu.mt_edges = {pdg: arrays[f"edges/{pdg}/mt"] for pdg in meta["pdgs"]}
        emu.models = {}
        for pdg, name in meta["models"]:
            prefix = f"model/{pdg}/{name}/"
            state = {
                key[len(prefix) :]: value
                for key, value in arrays.items()
                if key.startswith(prefix)
            }
            emu.models[(pdg, name)] = PCAGP.from_state(state)
        return emu
