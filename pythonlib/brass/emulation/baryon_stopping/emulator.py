import numpy as np

from ..base import Emulator
from .data import BaryonStoppingData

OBSERVABLES = ("dNdxF", "meanpT_xF", "dNdy", "meanmT_y")


class BaryonStoppingEmulator(Emulator):
    """Emulates BaryonStopping-analysis observables as a function of model parameters.

    Observables, each in the bins of the analysis:

    ``"dNdxF"``
        dN/dx_F per event;
    ``"meanpT_xF"``
        mean p_T versus x_F;
    ``"dNdy"``
        dN/dy per event;
    ``"meanmT_y"``
        mean m_T versus rapidity (subtract the mass for <m_T> - m).

    Mean values are emulated directly (not as a ratio of emulated spectra),
    which needs fewer PCA components and gives a Gaussian uncertainty. They
    are emulated in the bins that hold particles in every training run, so
    for rare hadrons the x_F range is narrower; :meth:`axis` gives the bins.

    Combine it with a :class:`~brass.emulation.bulk.BulkEmulator` of the same
    scan in one :class:`~brass.emulation.EmulatorLikelihood` with
    ``add(..., emulator=...)``.
    """

    OBSERVABLES = OBSERVABLES
    data_class = BaryonStoppingData

    def __init__(self, data, parameter_paths, observables=("dNdxF", "meanpT_xF"), **kwargs):
        super().__init__(data, parameter_paths, observables, **kwargs)
        self._pdgs = list(self.data.pdgs)
        self.axes = {}

    @property
    def pdgs(self):
        return self._pdgs

    def _fitted(self, pdg, name):
        self.axes[(pdg, name)] = self.data.axis(pdg, name)

    def axis(self, pdg, observable):
        """Bin centres of a fitted observable (x_F or y)."""
        if (pdg, observable) in self.axes:
            return self.axes[(pdg, observable)]
        if pdg not in self.pdgs:
            raise KeyError(f"no emulator for PDG {pdg}")
        raise ValueError(f"{observable!r} for PDG {pdg} is not fitted")

    def _save_arrays(self):
        return {f"axis/{pdg}/{name}": axis for (pdg, name), axis in self.axes.items()}

    def _load_arrays(self, arrays, meta):
        self._pdgs = list(meta["pdgs"])
        self.axes = {
            (pdg, name): np.asarray(arrays[f"axis/{pdg}/{name}"])
            for pdg, name in meta["models"]
        }
