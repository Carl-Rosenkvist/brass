# dndydmt_py.py
import numpy as np
import brass as br
from pathlib import Path
from brass import HistND


class Dndydmt:
    def __init__(self, y_edges, mt_edges, track_pdgs=None):
        self.y_edges = np.asarray(y_edges)
        self.mt_edges = np.asarray(mt_edges)
        # HistND expects a list of edges per dimension
        self.incl = HistND([self.mt_edges, self.y_edges])
        self.per_pdg = {}
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

    def merge_from(self, other, opts):
        self.incl.merge_(other.incl)
        for k, H in other.per_pdg.items():
            self.per_pdg.setdefault(k, HistND([self.mt_edges, self.y_edges]))
            self.per_pdg[k].merge_(H)
        self.n_events += getattr(other, "n_events", 0)

    def finalize(self, opts):
        dy = np.diff(self.y_edges)[0]
        dmt = np.diff(self.mt_edges)[0]
        n_ev = max(int(self.n_events), 1)
        norm = n_ev * dy * dmt
        self.incl.counts /= norm
        for H in self.per_pdg.values():
            H.counts /= norm

    def _fmt_val(self, v):
        return f"{round(v, 3):g}" if isinstance(v, float) else str(v)

    def _label_from_keys(self, keys: dict) -> str:
        parts = [f"{k}-{self._fmt_val(keys[k])}" for k in sorted(keys)]
        return "_".join(parts).replace("/", "-")

    def save(self, out_dir, keys, opts):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self._label_from_keys(keys)
        path = out_dir / f"dndydmt_{tag}.npz"

        np.savez(
            path,
            H_inclusive=self.incl.counts,
            y_edges=self.y_edges,
            mt_edges=self.mt_edges,
            n_events=int(self.n_events),
            **{f"pdg_{k}": H.counts for k, H in self.per_pdg.items()},
            keys=keys,
            analysis_name=getattr(self, "name", "dndydmt"),
            version=getattr(self, "smash_version", None),
        )


# --- Register analysis ---
edges_y = np.linspace(-4, 4, 31)
edges_mt = np.linspace(0.0, 3.5, 31)

br.register_python_analysis("dndydmt", lambda: Dndydmt(edges_y, edges_mt), {})
