# dndydmt_unwounded.py
import numpy as np
import brass as br
from brass import HistND


class DndydmtUnwounded:
    """
    dN/dy/dmT using mT = sqrt(E^2 - pz^2),
    grouped by number of UNWOUNDED nucleons (ncoll == 0),
    tracking only selected PDGs.
    NO inclusive histograms.
    NO normalization (user normalizes later).
    """

    def __init__(self, y_edges, mt_edges, track_pdgs):
        self.y_edges = np.asarray(y_edges)
        self.mt_edges = np.asarray(mt_edges)

        # PDGs to track
        self.track = set(track_pdgs)

        # structure:
        # per_unwounded[k] = {
        #     "n_events": int,
        #     "per_pdg": { pdg → HistND }
        # }
        self.per_unwounded: dict[int, dict] = {}

        # nucleons for wounded/unwounded classification
        self.NUCLEONS = {2212, 2112}

    # ---------------------------------------------------------------------
    # BRASS callbacks
    # ---------------------------------------------------------------------

    def on_interaction_block(self, iblock, accessor, opts):
        pass

    def on_end_block(self, block, accessor, opts):
        pass

    def on_particle_block(self, block, accessor, opts):

        pairs = accessor.gather_block_arrays(block)
        cols = {k: v for k, v in pairs}

        E    = cols["p0"]
        pz   = cols["pz"]
        pdg  = cols["pdg"]
        ncoll = cols["ncoll"]

        # -----------------------------------------------------------------
        # 1. Count UNWOUNDED nucleons
        # -----------------------------------------------------------------
        unwounded = np.logical_and(
            np.isin(pdg, list(self.NUCLEONS)),
            ncoll == 0
        ).sum()
        k = int(unwounded)

        # Ensure dictionary initialized
        if k not in self.per_unwounded:
            self.per_unwounded[k] = {"n_events": 0, "per_pdg": {}}

        # Count event in this unwounded class
        self.per_unwounded[k]["n_events"] += 1

        # -----------------------------------------------------------------
        # 2. Compute rapidity and mT
        # -----------------------------------------------------------------
        msk = E > np.abs(pz)
        if not msk.any():
            return

        E, pz, pdg = E[msk], pz[msk], pdg[msk]

        # rapidity
        y = 0.5 * np.log((E + pz) / (E - pz))

        # mT from E and pz:
        mt = np.sqrt(np.maximum(E*E - pz*pz, 0.0))

        # -----------------------------------------------------------------
        # 3. Fill PDG-specific histograms
        # -----------------------------------------------------------------
        present = np.intersect1d(np.unique(pdg), np.fromiter(self.track, dtype=int))

        for val in present:
            val = int(val)
            sel = (pdg == val)

            pdg_map = self.per_unwounded[k]["per_pdg"]
            H = pdg_map.setdefault(
                val,
                HistND([self.mt_edges, self.y_edges])
            )
            H.fill(mt, y, mask=sel)

    # ---------------------------------------------------------------------
    # Merging API
    # ---------------------------------------------------------------------

    def to_state_dict(self):
        return {
            "per_unwounded": self.per_unwounded
        }

    def finalize(self, results):
        """
        DO NOT normalize.
        User will normalize using per_unwounded[k]["n_events"] externally.
        Only merging happens in brass.
        """
        pass  # Intentionally empty


# -------------------------------------------------------------------------
# Register with brass
# -------------------------------------------------------------------------

edges_y = np.linspace(-4, 4, 31)
edges_mt = np.linspace(0.0, 3.5, 31)

TRACK_PDGS = [
    2212, -2212, 211, -211, 321, -321,
    3122, -3122, 3212, -3212,
    3312, -3312, 3322, -3322, 3334, -3334,
]

br.register_python_analysis(
    "DndydmtUnwounded",
    lambda: DndydmtUnwounded(edges_y, edges_mt, TRACK_PDGS),
    {},
)
