import numpy as np
import brass as br
from brass import HistND


class Dndydmt:
    def __init__(self, y_edges, mt_edges, track_pdgs=None):
        self.y_edges = np.asarray(y_edges)
        self.mt_edges = np.asarray(mt_edges)
        self.per_pdg: dict[int, HistND] = {}
        self.track = set(track_pdgs or [])
        self.n_events = 0

    def on_interaction_block(self, iblock, accessor, opts):
        pass

    def on_end_block(self, block, accessor, opts):
        pass

    def on_particle_block(self, block, accessor, opts):
        self.n_events += 1
        pairs = accessor.gather_block_arrays(block)
        cols = {k: v for k, v in pairs}
        E = cols["p0"]
        pz = cols["pz"]
        pdg = cols["pdg"]

        msk = E > np.abs(pz)
        if not msk.any():
            return

        E = E[msk]
        pz = pz[msk]
        pdg = pdg[msk]

        mt = np.sqrt(E**2 - pz**2)
        y = 0.5 * np.log((E + pz) / (E - pz))

        if self.track:
            present_tracked = np.intersect1d(
                np.unique(pdg), np.fromiter(self.track, dtype=int)
            )
            for val in present_tracked:
                sel = pdg == val
                H = self.per_pdg.setdefault(
                    int(val), HistND([self.mt_edges, self.y_edges])
                )
                H.fill(mt, y, mask=sel)

    def to_state_dict(self):
        return {
            "n_events": int(self.n_events),
            "per_pdg": dict(self.per_pdg),
        }

    def finalize(self, results):
        dy = np.diff(self.y_edges)[0]
        dmt = np.diff(self.mt_edges)[0]

        for meta_key, analyses in results.items():
            d = analyses.get("dndydmt")
            if d is None:
                continue

            n_ev = max(int(d.get("n_events", 0)), 1)
            norm = n_ev * dy * dmt

            for H in d.get("per_pdg", {}).values():
                if isinstance(H, HistND):
                    H.counts /= norm


edges_y = np.linspace(-4, 4, 31)
edges_mt = np.linspace(0.0, 3.5, 31)

br.register_python_analysis(
    "dndydmt",
    lambda: Dndydmt(
        edges_y,
        edges_mt,
        [
            2212,
            -2212,
            211,
            -211,
            321,
            -321,
            3122,
            -3122,
            3212,
            -3212,
            3312,
            -3312,
            3322,
            -3322,
            3334,
            -3334,
        ],
    ),
    {},
)
