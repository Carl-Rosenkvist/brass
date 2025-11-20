# dndydmt_py.py
import numpy as np
import brass as br
from pathlib import Path
from brass import HistND
import pickle


class Dndydmt:
    def __init__(self, y_edges, mt_edges, track_pdgs=None):
        self.y_edges = np.asarray(y_edges)
        self.mt_edges = np.asarray(mt_edges)

        # HistND expects a list of edges per dimension
        self.incl = HistND([self.mt_edges, self.y_edges])
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
        E, pz, px, py, pdg = cols["p0"], cols["pz"], cols["px"], cols["py"], cols["pdg"]

        # avoid y NaN; clamp negative m^2
        msk = E > np.abs(pz)
        if not msk.any():
            return
        E, pz, px, py, pdg = E[msk], pz[msk], px[msk], py[msk], pdg[msk]

        pt = np.hypot(px, py)
        m2 = np.maximum(E * E - (px * px + py * py + pz * pz), 0.0)
        m = np.sqrt(m2)
        mt = np.hypot(pt, m)
        y = 0.5 * np.log((E + pz) / (E - pz))

        # inclusive histogram
        self.incl.fill(mt, y)

        # tracked pdgs
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

    # --- New API: state export for brass merging ---

    def to_state_dict(self):
        """Return picklable state for this analysis instance.

        brass will merge these dicts from different workers and pass
        the merged structure into `finalize(results)`.
        """
        return {
            "n_events": int(self.n_events),
            "incl": self.incl,
            "per_pdg": dict(self.per_pdg),
        }

    def finalize(self, results):
        """Post-merge normalization.

        `results` has the structure:
        {
          meta_key_1: {
            "dndydmt": {
               "n_events": ...,
               "incl": HistND,
               "per_pdg": {pdg: HistND, ...}
            },
            ...
          },
          meta_key_2: { ... },
          ...
        }
        """
        # bin widths (assumes uniform)
        dy = np.diff(self.y_edges)[0]
        dmt = np.diff(self.mt_edges)[0]

        for meta_key, analyses in results.items():
            d = analyses.get("dndydmt")
            if d is None:
                continue

            n_ev = max(int(d.get("n_events", 0)), 1)
            norm = n_ev * dy * dmt

            H_incl = d.get("incl")
            if isinstance(H_incl, HistND):
                H_incl.counts /= norm

            for H in d.get("per_pdg", {}).values():
                if isinstance(H, HistND):
                    H.counts /= norm
 
# --- Register analysis ---
edges_y = np.linspace(-4, 4, 31)
edges_mt = np.linspace(0.0, 3.5, 31)

br.register_python_analysis(
    "dndydmt",
    lambda: Dndydmt(
        edges_y,
        edges_mt,
        [
            2212, -2212,          # p, pbar
            211, -211,            # pi+, pi-
            321, -321,            # K+, K-
            3122, -3122,          # Lambda
            3212, -3212,          # Sigma0
            3312, -3312,          # Xi-
            3322, -3322,          # Xi0
            3334, -3334,          # Omega-
        ],
    ),
    {},
)
