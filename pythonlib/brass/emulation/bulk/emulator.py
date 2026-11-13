import json

import numpy as np

from ..pcagp import PCAGP
from .data import DataHandler
from .observables import (
    SPECTRA,
    XF_OBSERVABLES,
    Observables,
    observable_from_json,
    observable_to_json,
)

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

    :class:`XfSpectrum` (dN/dx_F) and :class:`MeanPtXf` (<p_T> vs x_F) are
    fitted directly on the (x_F, p_T) histograms of a Bulk analysis run with
    ``--xf-bins``; give them in ``observables`` with the data's x_F bins.

    From a fitted ``"hist2d"``, :class:`MtSpectrum` (mT in a rapidity window)
    is derived as well. Pass any of these as ``observable`` to
    :meth:`predict_observable` or :meth:`~brass.emulation.EmulatorLikelihood.add`.

    ``mt_max`` and ``mt_group`` reduce the histograms before fitting (see
    :meth:`DataHandler.reduce`): crop at ``mt_max`` and merge ``mt_group``
    neighbouring mT bins. This makes a ``"hist2d"`` emulator much smaller.

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
        mt_max=None,
        mt_group=1,
    ):
        self.data = (
            data
            if isinstance(data, DataHandler)
            else DataHandler(data=data, parameter_paths=parameter_paths)
        )
        if mt_max is not None or mt_group != 1:
            self.data = self.data.reduce(mt_max=mt_max, mt_group=mt_group)

        observables = (observables,) if isinstance(observables, str) else observables
        unknown = [
            o
            for o in observables
            if not (o in OBSERVABLES or isinstance(o, XF_OBSERVABLES))
        ]
        if unknown:
            raise ValueError(
                f"unknown observables {unknown}; choose from {OBSERVABLES}, "
                "XfSpectrum or MeanPtXf"
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
                        f"{pdg:>7} {name!s:<7} {model.n_kept:3d} components "
                        f"({100 * model.explained_fraction:.1f} % of variance)"
                    )
        return self

    # -- observables ----------------------------------------------------------

    def observables(self, pdg):
        return Observables(self.y_edges[pdg], self.mt_edges[pdg])

    def axis(self, pdg, observable):
        """Bin centres of an observable (None for the total yield)."""
        if isinstance(observable, XF_OBSERVABLES):
            return observable.axis()
        obs = self.observables(pdg)
        if isinstance(observable, SPECTRA):
            return observable.build(obs)[0]
        return {"dNdy": obs.y, "dNdmt": obs.mt, "yield": None}.get(observable)

    def _derivation(self, pdg, observable, available):
        """(source, A) with ``observable = A @ source`` for a source in ``available``.

        A is None when ``observable`` itself is available.
        """
        if observable in available:
            return observable, None
        if pdg not in self.y_edges:
            raise KeyError(f"no emulator for PDG {pdg}")

        obs = self.observables(pdg)
        if isinstance(observable, SPECTRA):
            derived = {"hist2d": lambda: observable.build(obs)[1]}
        else:
            ny, nmt = len(obs.dy), len(obs.dmt)
            derived = {
                "dNdy": {"hist2d": lambda: np.kron(np.eye(ny), obs.dmt[None, :])},
                "dNdmt": {"hist2d": lambda: np.kron(obs.dy[None, :], np.eye(nmt))},
                "yield": {
                    "dNdy": lambda: obs.dy[None, :],
                    "dNdmt": lambda: obs.dmt[None, :],
                    "hist2d": lambda: np.kron(obs.dy, obs.dmt)[None, :],
                },
            }.get(observable, {})

        for source, matrix in derived.items():
            if source in available:
                return source, matrix()
        raise ValueError(
            f"{observable!r} for PDG {pdg} cannot be obtained from the fitted "
            f"observables {self.observable_names}"
        )

    def linear_model(self, pdg, observable):
        """Fitted model and matrix A with ``observable = A @ model output``.

        A is None when the observable is emulated directly. Otherwise it is
        derived linearly from another fitted observable (e.g. dN/dy from the
        2D histogram, the yield from dN/dy, or a pT spectrum from the 2D
        histogram).
        """
        if pdg in self.y_edges and not self.models:
            raise RuntimeError("emulator is not fitted; call fit() first")
        fitted = [name for p, name in self.models if p == pdg]
        source, A = self._derivation(pdg, observable, fitted)
        return self.models[(pdg, source)], A

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

        ``observables`` may include derived ones (e.g. ``"dNdy"`` or a
        :class:`MtSpectrum` from a ``"hist2d"`` emulator); they are then
        predicted from the refitted source model, as in the likelihood.
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
            refitted = {}
            for name in observables or self.observable_names:
                source, A = self._derivation(pdg, name, self.observable_names)
                if source not in refitted:
                    Y = self.data.observable(pdg, source)
                    refitted[source] = self._new_model().fit(X[~test], Y[~test])
                truth = self.data.observable(pdg, name)[test]
                mean, std = refitted[source].predict(X[test], A, return_std=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = (mean - truth) / std
                results[(pdg, name)] = {
                    "parameters": X[test],
                    "truth": truth,
                    "mean": mean,
                    "std": std,
                    "residuals": z,
                }
        return results

    # -- persistence ----------------------------------------------------------

    def save(self, path):
        """Save the fitted emulator (without training data) to a .npz file."""
        models = list(self.models.items())
        meta = {
            "format": 2,
            "parameter_names": self.parameter_names,
            "observables": [observable_to_json(o) for o in self.observable_names],
            "log_parameters": self.log_parameters,
            "settings": self.settings,
            "models": [[int(pdg), observable_to_json(name)] for (pdg, name), _ in models],
            "pdgs": [int(pdg) for pdg in self.pdgs],
        }
        arrays = {"meta": np.array(json.dumps(meta))}
        for pdg in self.pdgs:
            arrays[f"edges/{pdg}/y"] = self.y_edges[pdg]
            arrays[f"edges/{pdg}/mt"] = self.mt_edges[pdg]
        for i, (_, model) in enumerate(models):
            for key, value in model.state().items():
                arrays[f"model/{i}/{key}"] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}

        meta = json.loads(str(arrays["meta"]))
        emu = cls.__new__(cls)
        emu.data = None
        emu.parameter_names = meta["parameter_names"]
        emu.observable_names = [observable_from_json(o) for o in meta["observables"]]
        emu.log_parameters = meta["log_parameters"]
        emu.settings = meta["settings"]
        emu.y_edges = {pdg: arrays[f"edges/{pdg}/y"] for pdg in meta["pdgs"]}
        emu.mt_edges = {pdg: arrays[f"edges/{pdg}/mt"] for pdg in meta["pdgs"]}
        emu.models = {}
        for i, (pdg, name) in enumerate(meta["models"]):
            # Format 1 stored models under their (pdg, name), format 2 by index.
            prefix = f"model/{pdg}/{name}/" if meta["format"] == 1 else f"model/{i}/"
            state = {
                key[len(prefix) :]: value
                for key, value in arrays.items()
                if key.startswith(prefix)
            }
            emu.models[(pdg, observable_from_json(name))] = PCAGP.from_state(state)
        return emu
