import numpy as np

class Hist2D:
    def __init__(self, x_edges, y_edges):
        self.x_edges = np.asarray(x_edges)
        self.y_edges = np.asarray(y_edges)
        self.nx = len(self.x_edges) - 1
        self.ny = len(self.y_edges) - 1
        self.H = np.zeros((self.nx, self.ny), dtype=float)

    @staticmethod
    def _inbin_idx(edges, vals):
        # returns integer bin indices, -1 for underflow, nbins for overflow
        return np.searchsorted(edges, vals, side="right") - 1

    def fill(self, x, y, mask=None, weights=None):
        """
        Incremental fill with raw (x,y) values.
        """
        if mask is not None:
            x, y = x[mask], y[mask]
            if weights is not None:
                weights = weights[mask]

        bx = self._inbin_idx(self.x_edges, x)
        by = self._inbin_idx(self.y_edges, y)
        ok = (bx >= 0) & (bx < self.nx) & (by >= 0) & (by < self.ny)
        if not ok.any():
            return

        bx, by = bx[ok], by[ok]
        if weights is None:
            # readable, no “flat” indices needed
            np.add.at(self.H, (bx, by), 1.0)
        else:
            np.add.at(self.H, (bx, by), weights[ok])

    def fill_binned(self, bx, by, mask=None, weights=None):
        """
        Incremental fill if you already computed bin indices.
        """
        if mask is not None:
            bx, by = bx[mask], by[mask]
            if weights is not None:
                weights = weights[mask]
        ok = (bx >= 0) & (bx < self.nx) & (by >= 0) & (by < self.ny)
        if not ok.any():
            return
        if weights is None:
            np.add.at(self.H, (bx[ok], by[ok]), 1.0)
        else:
            np.add.at(self.H, (bx[ok], by[ok]), weights[ok])

    def merge_(self, other):
        self.H += other.H

    def normalized_copy(self, norm):
        out = Hist2D(self.x_edges, self.y_edges)
        out.H = self.H / float(norm)
        return out
