"""General PCA + Gaussian-process emulator of simulation outputs."""

import json
import time
from collections.abc import Mapping

import numpy as np

from .gp import fit_gps
from .pcagp import PCAGP


class Emulator:
    """Emulates outputs of a model as functions of its parameters.

    Give the parameters of the training runs, add outputs (any array with
    one row per run, e.g. dN/dy in bins) under a key of your choice, fit,
    and predict::

        # X: (n_runs, n_parameters), Y: (n_runs, n_bins)
        emu = Emulator(X, names=["Sqrtsnn", "Strange_Supp"])
        emu.add("p dNdy", Y, bins=y_edges)
        emu.fit()

        mean, std = emu.predict({"Sqrtsnn": 17.3, "Strange_Supp": 0.15}, "p dNdy",
                                return_std=True)
        y = emu.axis("p dNdy")

    Each output gets its own principal-component analysis; one GP (Matérn
    5/2 kernel, a length scale per parameter, fitted noise) is fitted per
    kept component, all GPs of all outputs in parallel. The uncertainty
    combines the GP variances with the covariance of the discarded
    components.

    Parameters
    ----------
    parameters
        ``{name: values}`` with one value per run, or an array of shape
        (n_runs, n_parameters) together with ``names``.
    n_components
        Number of PCA components per output (int), or the fraction of the
        variance to keep (float in (0, 1)).
    log_parameters
        Names of parameters that are emulated in ``log(x)``.
    noise
        Fit a noise term with each GP (recommended for Monte Carlo output).
    n_restarts, random_state
        Extra random starts of the hyperparameter search, and their seed.
    """

    def __init__(
        self,
        parameters,
        names=None,
        *,
        n_components=0.99,
        log_parameters=(),
        noise=True,
        n_restarts=0,
        random_state=None,
    ):
        if isinstance(parameters, Mapping):
            if names is not None:
                raise TypeError("the names of the parameters are the keys of the dict")
            names = list(parameters)
            columns = [np.asarray(parameters[n], dtype=float).ravel() for n in names]
            if len({len(c) for c in columns}) > 1:
                raise ValueError("every parameter needs one value per run")
            X = np.column_stack(columns) if columns else np.empty((0, 0))
        else:
            X = np.asarray(parameters, dtype=float)
            if X.ndim != 2 or names is None or len(names) != X.shape[1]:
                raise ValueError(
                    "give an array (n_runs, n_parameters) and one name per column"
                )
            names = list(names)
        if not names:
            raise ValueError("no parameters")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")
        if not np.all(np.isfinite(X)):
            raise ValueError("parameters must be finite")
        unknown = set(log_parameters) - set(names)
        if unknown:
            raise ValueError(
                f"log_parameters not among the parameters: {sorted(unknown)}"
            )

        self.parameter_names = names
        self.parameters = X
        self.log_parameters = list(log_parameters)
        self.settings = {
            "n_components": n_components,
            "noise": noise,
            "n_restarts": n_restarts,
            "random_state": random_state,
        }
        self.models = {}
        self._training = {}
        self._bins = {}

    def __repr__(self):
        state = (
            "fitted"
            if self.models and len(self.models) == len(self._bins)
            else "not fitted"
        )
        return f"Emulator({len(self.parameter_names)} parameters, outputs {self.keys}, {state})"

    # -- outputs ----------------------------------------------------------------------

    @property
    def keys(self):
        """Keys of the outputs, in the order they were added."""
        return list(self._bins)

    def add(self, key, values, bins=None):
        """Add an output to emulate; returns the emulator.

        ``values`` has one row per training run (a 1D array is one value per
        run). ``bins`` are the bin edges, or ``(lo, hi)``; they give
        :meth:`axis` and let
        :class:`EmulatorLikelihood` interpolate to data points. ``key`` is a
        string, an integer (e.g. a PDG code) or a tuple of them.
        """
        key = _key(key)
        if key in self._bins:
            raise ValueError(f"output {key!r} was already added")
        Y = np.asarray(values, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim != 2 or len(Y) != len(self.parameters):
            raise ValueError(
                f"values of {key!r} need one row per run ({len(self.parameters)}), "
                f"not shape {Y.shape}"
            )
        if not np.all(np.isfinite(Y)):
            raise ValueError(f"values of {key!r} are not all finite")
        self._training[key] = Y
        self._bins[key] = None if bins is None else _bins(bins, Y.shape[1], key)
        return self

    def axis(self, key):
        """Bin centres of an output (None if it was added without bins)."""
        bins = self.bins(key)
        return None if bins is None else 0.5 * (bins[0] + bins[1])

    def bins(self, key):
        """``(lo, hi)`` bin edges of an output (None if it was added without bins)."""
        key = _key(key)
        if key not in self._bins:
            raise KeyError(f"no output {key!r}; the outputs are {self.keys}")
        return self._bins[key]

    # -- fitting ----------------------------------------------------------------------

    def _new_model(self):
        return PCAGP(
            n_components=self.settings["n_components"],
            log_parameters=[self.parameter_names.index(p) for p in self.log_parameters],
            noise=self.settings["noise"],
            n_restarts=self.settings["n_restarts"],
            random_state=self.settings["random_state"],
        )

    def _fit_models(self, X, outputs, n_jobs):
        """PCA per output, then all their GPs in one parallel batch."""
        models, tasks, owners = {}, [], []
        for key, Y in outputs.items():
            model = self._new_model()
            new = model.prepare(X, Y)
            models[key] = model
            tasks += new
            owners += [key] * len(new)
        settings = {
            k: self.settings[k] for k in ("noise", "n_restarts", "random_state")
        }
        for key, gp in zip(owners, fit_gps(tasks, n_jobs=n_jobs, **settings)):
            models[key].gps.append(gp)
        return models

    def fit(self, n_jobs=-1, verbose=False):
        """Fit every output; ``n_jobs`` processes (-1: all cores)."""
        if self.parameters is None:
            raise RuntimeError("a loaded emulator has no training data to fit")
        if not self._training:
            raise RuntimeError("add outputs before fitting")
        start = time.perf_counter()
        self.models = self._fit_models(self.parameters, self._training, n_jobs)
        if verbose:
            for key, model in self.models.items():
                print(
                    f"{key!s:>20}  {model.n_kept:3d} components "
                    f"({100 * model.explained_fraction:.1f} % of variance)"
                )
            n_gps = sum(m.n_kept for m in self.models.values())
            print(f"{n_gps} GPs fitted in {time.perf_counter() - start:.0f} s")
        return self

    # -- prediction -------------------------------------------------------------------

    def inputs(self, theta):
        """Parameter array (n, n_parameters) from an array or ``{name: value(s)}``."""
        names = self.parameter_names
        if isinstance(theta, Mapping):
            missing = [n for n in names if n not in theta]
            unknown = [n for n in theta if n not in names]
            if missing or unknown:
                raise KeyError(f"parameters missing: {missing}, unknown: {unknown}")
            columns = np.broadcast_arrays(
                *(np.atleast_1d(np.asarray(theta[n], dtype=float)) for n in names)
            )
            return np.column_stack([c.ravel() for c in columns])
        X = np.atleast_2d(np.asarray(theta, dtype=float))
        if X.ndim != 2 or X.shape[1] != len(names):
            raise ValueError(
                f"theta needs {len(names)} columns, in this order: {names}"
            )
        return X

    def model(self, key):
        """The fitted :class:`~brass.emulation.PCAGP` of an output."""
        key = _key(key)
        if key not in self.models:
            if key in self._bins:
                raise RuntimeError(f"output {key!r} is not fitted; call fit()")
            raise KeyError(f"no output {key!r}; the outputs are {self.keys}")
        return self.models[key]

    def predict(self, theta, key=None, A=None, *, return_std=False, return_cov=False):
        """Emulated output at parameters ``theta``, shape (n, n_bins).

        ``theta``: array (n, n_parameters), one point, or ``{name: value(s)}``.
        Returns the mean, plus the standard deviation or covariance if asked.
        ``A`` maps the output linearly (``A @ output``, e.g. a sum of bins),
        with its uncertainty. Without ``key``, returns ``{key: result}`` for
        all outputs.
        """
        X = self.inputs(theta)
        if key is None:
            if A is not None:
                raise TypeError("A needs a key")
            return {
                k: self.predict(X, k, return_std=return_std, return_cov=return_cov)
                for k in self.models
            }
        return self.model(key).predict(
            X, A, return_std=return_std, return_cov=return_cov
        )

    # -- validation -------------------------------------------------------------------

    def validate(self, holdout=0.2, random_state=0, n_jobs=-1):
        """Refit without a random part of the runs and predict those runs.

        Returns ``{key: result}``; result holds the held-out ``parameters``,
        simulated ``truth``, emulated ``mean`` and ``std``, and the
        standardised residuals ``(mean - truth) / std``, which are roughly
        standard normal for a well-calibrated emulator.

        All runs at one parameter point (e.g. the same point at several
        energies) are held out together: parameters with few distinct values
        (at most 10 % of the runs) are taken as a grid.
        """
        if self.parameters is None:
            raise RuntimeError("a loaded emulator has no training data to validate")
        X = self.parameters
        grid = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) <= 0.1 * len(X)]
        keys = np.delete(X, grid, axis=1) if len(grid) < X.shape[1] else X
        _, point_of_run = np.unique(keys, axis=0, return_inverse=True)
        point_of_run = point_of_run.ravel()
        n_points = point_of_run.max() + 1

        rng = np.random.default_rng(random_state)
        n_test = min(max(1, round(holdout * n_points)), n_points - 1)
        test = np.isin(point_of_run, rng.choice(n_points, size=n_test, replace=False))

        train = {key: Y[~test] for key, Y in self._training.items()}
        models = self._fit_models(X[~test], train, n_jobs)
        results = {}
        for key, model in models.items():
            truth = self._training[key][test]
            mean, std = model.predict(X[test], return_std=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (mean - truth) / std
            results[key] = {
                "parameters": X[test],
                "truth": truth,
                "mean": mean,
                "std": std,
                "residuals": z,
            }
        return results

    # -- persistence ------------------------------------------------------------------

    def save(self, path):
        """Save the fitted emulator (without training data) to a .npz file."""
        missing = [k for k in self.keys if k not in self.models]
        if missing:
            raise RuntimeError(f"outputs {missing} are not fitted; call fit()")
        meta = {
            "format": 2,
            "class": "Emulator",
            "parameter_names": self.parameter_names,
            "log_parameters": self.log_parameters,
            "settings": self.settings,
            "outputs": [json.dumps(k) for k in self.keys],
        }
        arrays = {"meta": np.array(json.dumps(meta))}
        for i, key in enumerate(self.keys):
            if self._bins[key] is not None:
                arrays[f"output/{i}/lo"], arrays[f"output/{i}/hi"] = self._bins[key]
            for name, value in self.models[key].state().items():
                arrays[f"output/{i}/model/{name}"] = value
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            arrays = {key: f[key] for key in f.files}
        meta = json.loads(str(arrays["meta"]))
        saved = meta.get("class", "BulkEmulator")
        if saved != "Emulator":
            raise TypeError(f"{path} holds a {saved}; load it with {saved}.load()")

        emu = cls.__new__(cls)
        emu.parameter_names = meta["parameter_names"]
        emu.parameters = None
        emu.log_parameters = meta["log_parameters"]
        emu.settings = meta["settings"]
        emu._training = {}
        emu._bins = {}
        emu.models = {}
        for i, encoded in enumerate(meta["outputs"]):
            key = _decode(json.loads(encoded))
            prefix = f"output/{i}/model/"
            state = {
                k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)
            }
            emu.models[key] = PCAGP.from_state(state)
            lo = arrays.get(f"output/{i}/lo")
            emu._bins[key] = None if lo is None else (lo, arrays[f"output/{i}/hi"])
        return emu


def _key(key):
    """Output key as int, str or tuple of them (JSON-serialisable)."""
    if isinstance(key, np.integer):
        return int(key)
    if isinstance(key, tuple):
        return tuple(_key(k) for k in key)
    if isinstance(key, (int, str)) and not isinstance(key, bool):
        return key
    raise TypeError(
        f"output keys are strings, integers or tuples of them (e.g. 2212 or "
        f"(2212, 'dNdy')), not {key!r}"
    )


def _decode(value):
    """Key from its JSON form (tuples are stored as lists)."""
    return tuple(_decode(v) for v in value) if isinstance(value, list) else value


def _bins(bins, n, key):
    """(lo, hi) from ``(lo, hi)`` or from n + 1 edges."""
    if isinstance(bins, tuple) and len(bins) == 2 and np.ndim(bins[0]) == 1:
        lo, hi = (np.asarray(b, dtype=float) for b in bins)
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or len(edges) != n + 1:
            raise ValueError(f"bins of {key!r}: give {n + 1} edges or (lo, hi)")
        lo, hi = edges[:-1], edges[1:]
    if lo.shape != (n,) or hi.shape != (n,):
        raise ValueError(f"bins of {key!r}: lo and hi need {n} values each")
    return lo, hi
