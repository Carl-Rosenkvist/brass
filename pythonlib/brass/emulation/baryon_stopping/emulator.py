from typing import ClassVar

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

    The binning can be changed (counts and summed p_T or m_T add exactly);
    each setting is a value for all particles or a dict per PDG:

    ``fold=True``
        add the bins at -x and +x (symmetric systems such as pp): twice the
        statistics, half the outputs; observables are then for x >= 0;
    ``min_count``
        merge neighbouring bins, outward from x = 0, until each holds at
        least this many particles in every training run. Bins stay narrow
        where statistics are high and widen where they are low, so the
        training data are less noisy and fewer PCA components are needed;
    ``xf_group``, ``xf_max``
        merge a fixed number of x_F bins; keep |x_F| <= xf_max.

    Mean values are emulated only in bins that hold particles in every
    training run. :meth:`axis` and :meth:`bins` give the bins.

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
        fold=False,
        min_count=None,
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
        self.fold = self._per_particle(fold, "fold")
        self.min_count = self._per_particle(min_count, "min_count")
        self.axes = {}
        self._bins = {}

    _DEFAULTS: ClassVar[dict] = {
        "xf_max": None,
        "xf_group": 1,
        "fold": False,
        "min_count": None,
    }

    def _per_particle(self, value, name):
        if not isinstance(value, dict):
            return {pdg: value for pdg in self._pdgs}
        unknown = set(value) - set(self._pdgs)
        if unknown:
            raise KeyError(f"{name} for PDGs that are not emulated: {sorted(unknown)}")
        return {pdg: value.get(pdg, self._DEFAULTS[name]) for pdg in self._pdgs}

    @property
    def pdgs(self):
        return self._pdgs

    def _observables_for(self, pdg):
        return self._per_pdg.get(pdg, [])

    def _binning(self, pdg):
        return {name: getattr(self, name)[pdg] for name in self._DEFAULTS}

    def _training(self, pdg, name):
        return self.data.observable(pdg, name, **self._binning(pdg))

    def _fitted(self, pdg, name):
        lo, hi = self.data.bins(pdg, name, **self._binning(pdg))
        self._bins[(pdg, name)] = (lo, hi)
        self.axes[(pdg, name)] = 0.5 * (lo + hi)

    def axis(self, pdg, observable):
        """Bin centres of a fitted observable (x_F or y)."""
        if (pdg, observable) in self.axes:
            return self.axes[(pdg, observable)]
        if pdg not in self.pdgs:
            raise KeyError(f"no emulator for PDG {pdg}")
        raise ValueError(f"{observable!r} for PDG {pdg} is not fitted")

    def bins(self, pdg, observable):
        """Lower and upper bin edges of a fitted observable."""
        self.axis(pdg, observable)  # same errors as axis()
        return self._bins[(pdg, observable)]

    def _save_arrays(self):
        arrays = {}
        for (pdg, name), (lo, hi) in self._bins.items():
            arrays[f"axis/{pdg}/{name}"] = self.axes[(pdg, name)]
            arrays[f"bins/{pdg}/{name}/lo"] = lo
            arrays[f"bins/{pdg}/{name}/hi"] = hi
        return arrays

    def _extra_meta(self):
        extra = {"observables": {str(p): names for p, names in self._per_pdg.items()}}
        for name in self._DEFAULTS:
            extra[name] = {str(p): v for p, v in getattr(self, name).items()}
        return extra

    def _load_arrays(self, arrays, meta):
        self._pdgs = list(meta["pdgs"])
        self.axes = {
            (pdg, name): np.asarray(arrays[f"axis/{pdg}/{name}"])
            for pdg, name in meta["models"]
        }
        self._bins = {}
        for pdg, name in meta["models"]:
            if f"bins/{pdg}/{name}/lo" in arrays:
                lo, hi = (
                    np.asarray(arrays[f"bins/{pdg}/{name}/{e}"]) for e in ("lo", "hi")
                )
            else:  # older files: equal bins around the stored centres
                x = self.axes[(pdg, name)]
                half = 0.5 * np.min(np.diff(x)) if len(x) > 1 else 0.0
                lo, hi = x - half, x + half
            self._bins[(pdg, name)] = (lo, hi)
        extra = meta.get("extra") or {}
        self._per_pdg = {
            int(p): v for p, v in extra.get("observables", {}).items()
        } or {pdg: [n for p, n in meta["models"] if p == pdg] for pdg in self._pdgs}
        for name, default in self._DEFAULTS.items():
            stored = extra.get(name, {})
            setattr(
                self, name, {pdg: stored.get(str(pdg), default) for pdg in self._pdgs}
            )
