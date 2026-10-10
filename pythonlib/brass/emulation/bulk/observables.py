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
