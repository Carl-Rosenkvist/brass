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

    ``observables`` is a list (the same observables for every particle) or a
    dict ``{pdg: [observables]}``; particles not in the dict are not
    emulated. Rare particles have too few entries for <p_T>(x_F), so choose
    per particle::

        BaryonStoppingEmulator.from_file(
            "baryon-stopping.pkl", paths,
            observables={2212: ["dNdxF", "meanpT_xF"], -2212: ["meanpT_xF"]},
            xf_max={-2212: 0.3},
            xf_group=2,
        )

    ``xf_group`` merges that many neighbouring x_F bins (counts and summed
    p_T add exactly) and ``xf_max`` keeps only bins within |x_F| <= xf_max;
    each is a number for all particles or a dict per PDG. Mean values are
    emulated only in bins that hold particles in every training run;
    :meth:`axis` gives the bins.

    Mean values are emulated directly (not as a ratio of emulated spectra),
    which needs fewer PCA components and gives a Gaussian uncertainty.
    Combine it with a :class:`~brass.emulation.bulk.BulkEmulator` of the same
    scan in one :class:`~brass.emulation.EmulatorLikelihood` with
    ``add(..., emulator=...)``.
    """

    OBSERVABLES = OBSERVABLES
    data_class = BaryonStoppingData

    def __init__(
        self,
        data,
        parameter_paths,
        observables=("dNdxF", "meanpT_xF"),
        *,
        xf_max=None,
        xf_group=1,
        **kwargs,
    ):
        per_pdg = None
        if isinstance(observables, dict):
            per_pdg = {
                int(pdg): [names] if isinstance(names, str) else list(names)
                for pdg, names in observables.items()
            }
            observables = list(
                dict.fromkeys(name for names in per_pdg.values() for name in names)
            )

        super().__init__(data, parameter_paths, observables, **kwargs)

        if per_pdg is None:
            per_pdg = {pdg: list(self.observable_names) for pdg in self.data.pdgs}
        unknown = set(per_pdg) - set(self.data.pdgs)
        if unknown:
            raise KeyError(f"PDGs not in the data: {sorted(unknown)}")
        self._per_pdg = per_pdg
        self._pdgs = list(per_pdg)
        self.xf_max = self._per_particle(xf_max, "xf_max")
        self.xf_group = self._per_particle(xf_group, "xf_group")
        self.axes = {}

    def _per_particle(self, value, name):
        if not isinstance(value, dict):
            return {pdg: value for pdg in self._pdgs}
        unknown = set(value) - set(self._pdgs)
        if unknown:
            raise KeyError(f"{name} for PDGs that are not emulated: {sorted(unknown)}")
        default = 1 if name == "xf_group" else None
        return {pdg: value.get(pdg, default) for pdg in self._pdgs}

    @property
    def pdgs(self):
        return self._pdgs

    def _observables_for(self, pdg):
        return self._per_pdg.get(pdg, [])

    def _binning(self, pdg):
        return {"xf_group": self.xf_group[pdg], "xf_max": self.xf_max[pdg]}

    def _training(self, pdg, name):
        return self.data.observable(pdg, name, **self._binning(pdg))

    def _fitted(self, pdg, name):
        self.axes[(pdg, name)] = self.data.axis(pdg, name, **self._binning(pdg))

    def axis(self, pdg, observable):
        """Bin centres of a fitted observable (x_F or y)."""
        if (pdg, observable) in self.axes:
            return self.axes[(pdg, observable)]
        if pdg not in self.pdgs:
            raise KeyError(f"no emulator for PDG {pdg}")
        raise ValueError(f"{observable!r} for PDG {pdg} is not fitted")

    def _save_arrays(self):
        return {f"axis/{pdg}/{name}": axis for (pdg, name), axis in self.axes.items()}

    def _extra_meta(self):
        return {
            "observables": {str(p): names for p, names in self._per_pdg.items()},
            "xf_max": {str(p): v for p, v in self.xf_max.items()},
            "xf_group": {str(p): v for p, v in self.xf_group.items()},
        }

    def _load_arrays(self, arrays, meta):
        self._pdgs = list(meta["pdgs"])
        self.axes = {
            (pdg, name): np.asarray(arrays[f"axis/{pdg}/{name}"])
            for pdg, name in meta["models"]
        }
        extra = meta.get("extra") or {}
        if "observables" in extra:
            self._per_pdg = {int(p): v for p, v in extra["observables"].items()}
            self.xf_max = {int(p): v for p, v in extra["xf_max"].items()}
            self.xf_group = {int(p): v for p, v in extra["xf_group"].items()}
        else:
            self._per_pdg = {
                pdg: [n for p, n in meta["models"] if p == pdg] for pdg in self._pdgs
            }
            self.xf_max = {pdg: None for pdg in self._pdgs}
            self.xf_group = {pdg: 1 for pdg in self._pdgs}
