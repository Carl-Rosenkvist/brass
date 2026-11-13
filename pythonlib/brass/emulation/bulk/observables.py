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
# "hist2d" emulator together with its covariance.
#
# x_F and pT observables are not derived here: within the (y, mT) bins they
# are too inaccurate (dN/dx_F off by 10-30 %, low-pT spectra by up to a
# factor of a few). Use the BaryonStopping analysis for x_F.


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


SPECTRA = (MtSpectrum,)
