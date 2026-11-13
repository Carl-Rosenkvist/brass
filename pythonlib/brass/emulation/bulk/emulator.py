import numpy as np

from ..base import Emulator
from .data import DataHandler
from .observables import SPECTRA, Observables

OBSERVABLES = ("dNdy", "dNdmt", "hist2d")


class BulkEmulator(Emulator):
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

    From a fitted ``"hist2d"``, :class:`MtSpectrum` (mT in a rapidity window)
    is derived as well; pass it as ``observable`` to
    :meth:`predict_observable` or :meth:`~brass.emulation.EmulatorLikelihood.add`.

    ``mt_max`` and ``mt_group`` reduce the histograms before fitting (see
    :meth:`DataHandler.reduce`): crop at ``mt_max`` and merge ``mt_group``
    neighbouring mT bins. This makes a ``"hist2d"`` emulator much smaller.

    Every parameter in ``parameter_paths`` is a GP input, including e.g. the
    collision energy, so one emulator covers an energy scan.
    """

    OBSERVABLES = OBSERVABLES
    data_class = DataHandler

    def __init__(
        self,
        data,
        parameter_paths,
        observables=("dNdy",),
        *,
        mt_max=None,
        mt_group=1,
        **kwargs,
    ):
        if not isinstance(data, DataHandler):
            data = DataHandler(data=data, parameter_paths=parameter_paths)
        if mt_max is not None or mt_group != 1:
            data = data.reduce(mt_max=mt_max, mt_group=mt_group)
        super().__init__(data, parameter_paths, observables, **kwargs)
        self.y_edges = dict(self.data.y_edges)
        self.mt_edges = dict(self.data.mt_edges)

    @property
    def pdgs(self):
        return list(self.y_edges)

    # -- observables ----------------------------------------------------------

    def observables(self, pdg):
        return Observables(self.y_edges[pdg], self.mt_edges[pdg])

    def axis(self, pdg, observable):
        """Bin centres of an observable (None for the total yield)."""
        obs = self.observables(pdg)
        if isinstance(observable, SPECTRA):
            return observable.build(obs)[0]
        return {"dNdy": obs.y, "dNdmt": obs.mt, "yield": None}.get(observable)

    def _derivation(self, pdg, observable, available):
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
        return super()._derivation(pdg, observable, available)

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

    # -- persistence ----------------------------------------------------------

    def _save_arrays(self):
        arrays = {}
        for pdg in self.pdgs:
            arrays[f"edges/{pdg}/y"] = self.y_edges[pdg]
            arrays[f"edges/{pdg}/mt"] = self.mt_edges[pdg]
        return arrays

    def _load_arrays(self, arrays, meta):
        self.y_edges = {pdg: arrays[f"edges/{pdg}/y"] for pdg in meta["pdgs"]}
        self.mt_edges = {pdg: arrays[f"edges/{pdg}/mt"] for pdg in meta["pdgs"]}
