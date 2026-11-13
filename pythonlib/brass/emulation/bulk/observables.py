from dataclasses import dataclass
from itertools import pairwise

import numpy as np


class Observables:
    def __init__(self, y_edges, mt_edges):

        self.y_edges = np.asarray(y_edges)
        self.mt_edges = np.asarray(mt_edges)

        self.dy = np.diff(self.y_edges)
        self.dmt = np.diff(self.mt_edges)

        self.y = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

        self.mt = 0.5 * (self.mt_edges[:-1] + self.mt_edges[1:])

    def dNdy(self, hist):

        return np.sum(
            hist * self.dmt,
            axis=-1,
        )

    def dNdmt(self, hist):

        return np.sum(
            hist * self.dy[:, None],
            axis=-2,
        )

    def yield_total(self, hist):

        return np.sum(hist * self.dy[:, None] * self.dmt[None, :])

    def y_overlap(self, y_min, y_max):
        """Length of the overlap of each rapidity bin with [y_min, y_max]."""
        if not y_min < y_max:
            raise ValueError(f"empty rapidity window [{y_min}, {y_max}]")
        if y_min < self.y_edges[0] - 1e-9 or y_max > self.y_edges[-1] + 1e-9:
            raise ValueError(
                f"rapidity window [{y_min}, {y_max}] is outside the histogram "
                f"[{self.y_edges[0]}, {self.y_edges[-1]}]"
            )
        lo = np.maximum(self.y_edges[:-1], y_min)
        hi = np.minimum(self.y_edges[1:], y_max)
        return np.clip(hi - lo, 0.0, None)


# -- spectra derived from the 2D histogram ------------------------------------
#
# Each spec gives the bin centres of the spectrum (``axis``) and a matrix A
# with ``spectrum = A @ hist.ravel()``, so it can be derived from a fitted
# "hist2d" emulator together with its covariance. Within a (y, mT) bin the
# density is taken as uniform.


@dataclass(frozen=True)
class MtSpectrum:
    """d²N/(dy dmT) averaged over the rapidity window [y_min, y_max].

    The axis is the mT bin centres. Bins that only partly overlap the window
    are weighted by the overlap.
    """

    y_min: float
    y_max: float

    def __str__(self):
        return f"mT[{self.y_min:g}<y<{self.y_max:g}]"

    def build(self, obs):
        weights = obs.y_overlap(self.y_min, self.y_max) / (self.y_max - self.y_min)
        return obs.mt, np.kron(weights[None, :], np.eye(len(obs.dmt)))


# -- spectra from the (x_F, p_T) histogram -------------------------------------
#
# Bulk fills these with --xf-bins. They are fitted directly by the emulator
# (they are not derived from the (y, mT) histogram, whose bins are too coarse
# for x_F: dN/dx_F would be off by 10-30 % and <p_T>(x_F) biased). ``edges``
# are the x_F bins of the data; each must be an edge of the stored binning.


def _xf_groups(edges, xf_edges):
    """Index ranges of the stored x_F bins that make up each bin of ``edges``."""
    index = []
    for edge in edges:
        match = np.flatnonzero(np.isclose(xf_edges, edge, rtol=0, atol=1e-9))
        if len(match) == 0:
            raise ValueError(
                f"x_F edge {edge:g} is not an edge of the stored binning "
                f"({len(xf_edges) - 1} bins in {xf_edges[0]:g}..{xf_edges[-1]:g})"
            )
        index.append(int(match[0]))
    return list(pairwise(index))


def _check_edges(self):
    edges = tuple(float(e) for e in self.edges)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("x_F edges must be increasing with at least two entries")
    object.__setattr__(self, "edges", edges)


@dataclass(frozen=True)
class XfSpectrum:
    """dN/dx_F per event in the x_F bins ``edges``.

    Particles outside the p_T range of the histogram are not counted.
    """

    edges: tuple

    __post_init__ = _check_edges

    def __str__(self):
        return f"dN/dxF[{self.edges[0]:g}..{self.edges[-1]:g}, {len(self.edges) - 1} bins]"

    def axis(self):
        e = np.asarray(self.edges)
        return 0.5 * (e[:-1] + e[1:])

    def values(self, hist, xf_edges, pt_edges):
        """From d²N/(dx_F dp_T) of shape (n_runs, n_xf, n_pt)."""
        counts = hist @ np.diff(pt_edges) * np.diff(xf_edges)  # per event
        return np.stack(
            [counts[:, a:b].sum(axis=1) for a, b in _xf_groups(self.edges, xf_edges)],
            axis=1,
        ) / np.diff(self.edges)


@dataclass(frozen=True)
class MeanPtXf:
    """Mean p_T in the x_F bins ``edges``.

    Computed from the p_T bin centres of the histogram (with 0.1 GeV bins
    the bias is below a tenth of the statistical error of 10⁶ events).
    Every training run needs particles in every bin.
    """

    edges: tuple

    __post_init__ = _check_edges

    def __str__(self):
        return f"<pT>(xF)[{self.edges[0]:g}..{self.edges[-1]:g}, {len(self.edges) - 1} bins]"

    def axis(self):
        return XfSpectrum(self.edges).axis()

    def values(self, hist, xf_edges, pt_edges):
        pt = 0.5 * (pt_edges[:-1] + pt_edges[1:])
        weights = hist * np.diff(pt_edges)  # d N/dx_F per p_T bin
        n, s = [], []
        for a, b in _xf_groups(self.edges, xf_edges):
            w = weights[:, a:b].sum(axis=1)
            n.append(w.sum(axis=1))
            s.append(w @ pt)
        n, s = np.stack(n, axis=1), np.stack(s, axis=1)
        empty = np.argwhere(n <= 0)
        if len(empty):
            run, k = empty[0]
            raise ValueError(
                f"no particles in x_F bin [{self.edges[k]:g}, {self.edges[k + 1]:g}] "
                f"of run {run}; use a narrower x_F range"
            )
        return s / n


# Derived linearly from a fitted "hist2d" model.
SPECTRA = (MtSpectrum,)
# Fitted directly on the (x_F, p_T) histogram.
XF_OBSERVABLES = (XfSpectrum, MeanPtXf)

_TYPES = {cls.__name__: cls for cls in (*SPECTRA, *XF_OBSERVABLES)}


def observable_to_json(observable):
    """JSON-serialisable form of an observable name or spec."""
    if isinstance(observable, str):
        return observable
    data = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in observable.__dict__.items()
    }
    return {"type": type(observable).__name__, **data}


def observable_from_json(data):
    if isinstance(data, str):
        return data
    data = dict(data)
    cls = _TYPES[data.pop("type")]
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
