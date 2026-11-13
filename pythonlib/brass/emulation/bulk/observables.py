from dataclasses import dataclass

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


@dataclass(frozen=True)
class PtSpectrum:
    """d²N/(dy dpT) averaged over the rapidity window [y_min, y_max].

    Obtained from the mT spectrum with d²N/(dy dpT) = pT/mT · d²N/(dy dmT),
    evaluated at the mT bin centres. The axis is pT = sqrt(mT² - mass²) of
    the bins whose centre lies above the mass.
    """

    y_min: float
    y_max: float
    mass: float

    def __str__(self):
        return f"pT[{self.y_min:g}<y<{self.y_max:g}, m={self.mass:g}]"

    def build(self, obs):
        _, A = MtSpectrum(self.y_min, self.y_max).build(obs)
        keep = obs.mt > self.mass
        if not keep.any():
            raise ValueError(f"no mT bin centre above the mass {self.mass}")
        pt = np.sqrt(obs.mt[keep] ** 2 - self.mass**2)
        return pt, (pt / obs.mt[keep])[:, None] * A[keep]


@dataclass(frozen=True)
class XfSpectrum:
    """dN/dx_F with x_F = 2 pz/√s = 2 mT sinh(y)/√s, binned in ``edges``.

    Each (y, mT) bin is split into ``n_sub`` × ``n_sub`` cells which are
    assigned to the x_F bins. Only particles inside the histogram's y and mT
    range contribute; choose ``edges`` that this range covers.
    """

    sqrts: float
    edges: tuple
    n_sub: int = 16

    def __post_init__(self):
        edges = tuple(float(e) for e in self.edges)
        if len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("x_F edges must be increasing with at least two entries")
        object.__setattr__(self, "edges", edges)

    def __str__(self):
        return f"xF[sqrts={self.sqrts:g}]"

    def build(self, obs):
        edges = np.asarray(self.edges)
        n = self.n_sub
        sub = (np.arange(n) + 0.5) / n
        y = obs.y_edges[:-1, None] + obs.dy[:, None] * sub  # (n_y, n)
        mt = obs.mt_edges[:-1, None] + obs.dmt[:, None] * sub  # (n_mt, n)
        xf = (
            2.0
            / self.sqrts
            * mt[None, :, None, :]
            * np.sinh(y[:, None, :, None])
        )  # (n_y, n_mt, n, n)

        n_y, n_mt = len(obs.dy), len(obs.dmt)
        cell = np.broadcast_to(
            np.arange(n_y * n_mt).reshape(n_y, n_mt, 1, 1), xf.shape
        ).ravel()
        k = np.searchsorted(edges, xf.ravel(), side="right") - 1
        inside = (k >= 0) & (k < len(edges) - 1)

        fraction = np.zeros((len(edges) - 1, n_y * n_mt))
        np.add.at(fraction, (k[inside], cell[inside]), 1.0 / n**2)

        area = np.kron(obs.dy, obs.dmt)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return centres, fraction * area[None, :] / np.diff(edges)[:, None]


SPECTRA = (MtSpectrum, PtSpectrum, XfSpectrum)
