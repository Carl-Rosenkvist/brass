import numpy as np

class Hist1D:
    def __init__(self, edges):
        self.edges = np.asarray(edges)
        self.nbins = len(self.edges) - 1
        self.H = np.zeros(self.nbins, dtype=float)

    @staticmethod
    def _bin_idx(edges, vals):
        return np.searchsorted(edges, vals, side="right") - 1

    def fill(self, x, mask=None, weights=None):
        if mask is not None:
            x = x[mask]
            if weights is not None:
                weights = weights[mask]

        b = self._bin_idx(self.edges, x)
        ok = (b >= 0) & (b < self.nbins)
        if not ok.any():
            return
        b = b[ok]
        if weights is None:
            np.add.at(self.H, b, 1.0)
        else:
            np.add.at(self.H, b, weights[ok])

    def merge_(self, other):
        self.H += other.H

    def normalized_copy(self, norm):
        out = Hist1D(self.edges)
        out.H = self.H / float(norm)
        return out
