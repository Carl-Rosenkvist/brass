"""Shared parts of the emulators of analysis results.

:class:`RunData` reads the model parameters of every run of a scan;
subclasses add the analysis-specific training data. :class:`Emulator` fits
one :class:`~brass.emulation.pcagp.PCAGP` per particle and observable and
handles prediction, validation and saving; subclasses define the
observables and their axes.
"""

import json
import pickle

import numpy as np

from .pcagp import PCAGP


class RunData:
    """Model parameters of every run of a scan, plus analysis data.

    ``data`` is a list of ``{"result": analysis, ...}`` (as written by
    ``brass`` or ``AnalysisRunner``). ``parameter_paths`` maps a parameter
    name to its path in the run's config; a string key inside a list of
    PYTHIA ``"key = value"`` settings is looked up by name.
    """

    def __init__(self, data, parameter_paths):
        self.parameter_paths = dict(parameter_paths)
        self.parameter_names = list(self.parameter_paths)

        parameters = []
        for run, result in enumerate(data):
            analysis = result["result"]
            if analysis.events <= 0:
                raise ValueError(f"run {run} ({analysis.folder_path}) has no events")
            parameters.append(self._extract_parameters(analysis))
            self._add_run(run, analysis)

        self.parameters = np.asarray(parameters, dtype=float)
        self._finish(len(parameters))

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

    # -- implemented by subclasses ------------------------------------------

    def _add_run(self, run, analysis):
        raise NotImplementedError

    def _finish(self, n_runs):
        pass

    @property
    def pdgs(self):
        raise NotImplementedError

    def observable(self, pdg, name):
        """Training values of an observable, shape (n_runs, n_outputs)."""
        raise NotImplementedError


def _pythia_setting(entries, key):
    """Value of ``key`` in a list of PYTHIA ``"key = value"`` strings."""
    for entry in entries:
        name, separator, value = entry.partition("=")
        if separator and name.strip() == key:
            return value.strip()
    raise KeyError(key)


def collect_runs(per_run, n_runs, what):
    """Stack per-PDG lists of per-run arrays; every PDG must be in every run."""
    missing = {pdg: n_runs - len(v) for pdg, v in per_run.items() if len(v) != n_runs}
    if missing:
        raise ValueError(f"PDGs missing in some runs of the {what} (pdg: runs): {missing}")
    return {pdg: np.stack(v) for pdg, v in per_run.items()}


class Emulator:
    """PCA + GP emulators of analysis observables, one per particle and observable.

    Every parameter in ``parameter_paths`` is a GP input, including e.g. the
    collision energy, so one emulator covers an energy scan. Subclasses set
    ``OBSERVABLES``, ``data_class`` and implement :meth:`axis` (and
    :meth:`_derivation` for observables derived linearly from fitted ones).
    """

    OBSERVABLES = ()
    data_class = RunData

    def __init__(
        self,
        data,
        parameter_paths,
        observables,
        n_components=0.99,
        log_parameters=(),
        noise=True,
        n_restarts=0,
        random_state=None,
        include_truncation=True,
    ):
        self.data = (
            data
            if isinstance(data, self.data_class)
            else self.data_class(data=data, parameter_paths=parameter_paths)
        )

        observables = (observables,) if isinstance(observables, str) else observables
        unknown = [o for o in observables if o not in self.OBSERVABLES]
        if unknown:
            raise ValueError(
                f"unknown observables {unknown}; choose from {self.OBSERVABLES}"
            )

        self.parameter_names = list(self.data.parameter_names)
        unknown = set(log_parameters) - set(self.parameter_names)
        if unknown:
            raise ValueError(f"log_parameters not in parameter_paths: {sorted(unknown)}")

        self.observable_names = list(observables)
        self.log_parameters = list(log_parameters)
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
        return cls(cls.data_class.from_file(path, parameter_paths), parameter_paths, **kwargs)

    # -- fitting --------------------------------------------------------------

    @property
    def pdgs(self):
        raise NotImplementedError

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
            for name in self._observables_for(pdg):
                model = self._new_model().fit(
                    self.data.parameters, self._training(pdg, name)
                )
                self.models[(pdg, name)] = model
                self._fitted(pdg, name)
                if verbose:
                    print(
                        f"{pdg:>7} {name:<9} {model.n_kept:3d} components "
                        f"({100 * model.explained_fraction:.1f} % of variance)"
                    )
        return self

    def _fitted(self, pdg, name):
        """Hook called after fitting (pdg, name)."""

    def _observables_for(self, pdg):
        """Observables fitted for a particle (all of them by default)."""
        return self.observable_names

    def _training(self, pdg, name):
        """Training values of an observable, shape (n_runs, n_outputs)."""
        return self.data.observable(pdg, name)

    # -- observables ----------------------------------------------------------

    def axis(self, pdg, observable):
        """Bin centres of an observable."""
        raise NotImplementedError

    def _derivation(self, pdg, observable, available):
        """(source, A) with ``observable = A @ source`` for a source in ``available``.

        A is None when ``observable`` itself is available.
        """
        if observable in available:
            return observable, None
        if pdg not in self.pdgs:
            raise KeyError(f"no emulator for PDG {pdg}")
        raise ValueError(
            f"{observable!r} for PDG {pdg} cannot be obtained from the fitted "
            f"observables {list(available)}"
        )

    def linear_model(self, pdg, observable):
        """Fitted model and matrix A with ``observable = A @ model output``.

        A is None when the observable is emulated directly. Otherwise it is
        derived linearly from another fitted observable.
        """
        if pdg in self.pdgs and not self.models:
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

    # -- validation -----------------------------------------------------------

    def validate(self, holdout=0.2, random_state=0, observables=None, pdgs=None):
        """Refit on a random subset of runs and predict the held-out runs.

        Returns ``{(pdg, observable): result}`` where result holds the
        held-out ``parameters``, simulated ``truth``, emulated ``mean`` and
        ``std``, and the standardised residuals ``(mean - truth) / std``. For
        a well-calibrated emulator these are roughly standard normal.

        All runs at a parameter point (e.g. the same point at several
        energies) are held out together. Derived observables are predicted
        from the refitted source model, as in the likelihood. With explicit
        ``observables`` and no ``pdgs``, particles without them are skipped.
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

        if pdgs is None:
            pdgs = self.pdgs
            if observables:
                usable = [p for p in pdgs if self._obtainable(p, observables)]
                # If no particle has them, the loop below raises the reason.
                pdgs = usable or pdgs[:1]

        results = {}
        for pdg in pdgs:
            refitted = {}
            available = self._observables_for(pdg)
            for name in observables or available:
                source, A = self._derivation(pdg, name, available)
                if source not in refitted:
                    Y = self._training(pdg, source)
                    refitted[source] = self._new_model().fit(X[~test], Y[~test])
                truth = self._training(pdg, name)[test]
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

    def _obtainable(self, pdg, observables):
        try:
            for name in observables:
                self._derivation(pdg, name, self._observables_for(pdg))
        except ValueError:
            return False
        return True

    # -- persistence ----------------------------------------------------------

    def _save_arrays(self):
        """Arrays besides the models needed after loading (e.g. bin edges)."""
        return {}

    def _load_arrays(self, arrays, meta):
        pass

    def _extra_meta(self):
        """JSON-serialisable settings needed after loading."""
        return {}

    def save(self, path):
        """Save the fitted emulator (without training data) to a .npz file."""
        meta = {
            "format": 1,
            "class": type(self).__name__,
            "parameter_names": self.parameter_names,
            "observables": self.observable_names,
            "log_parameters": self.log_parameters,
            "settings": self.settings,
            "models": [[int(pdg), name] for pdg, name in self.models],
            "pdgs": [int(pdg) for pdg in self.pdgs],
            "extra": self._extra_meta(),
        }
        arrays = {"meta": np.array(json.dumps(meta)), **self._save_arrays()}
        for (pdg, name), model in self.models.items():
            for key, value in model.state().items():
                arrays[f"model/{pdg}/{name}/{key}"] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}

        meta = json.loads(str(arrays["meta"]))
        saved = meta.get("class", "BulkEmulator")  # older files: always Bulk
        if saved != cls.__name__:
            raise TypeError(f"{path} holds a {saved}; load it with {saved}.load()")

        emu = cls.__new__(cls)
        emu.data = None
        emu.parameter_names = meta["parameter_names"]
        emu.observable_names = meta["observables"]
        emu.log_parameters = meta["log_parameters"]
        emu.settings = meta["settings"]
        emu._load_arrays(arrays, meta)
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
