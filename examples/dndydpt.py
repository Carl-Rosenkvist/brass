import numpy as np
import brass as br
from pathlib import Path

class Dndydpt:
    def __init__(self, y_edges, pt_edges, track_pdgs=None):
        self.y_edges  = np.asarray(y_edges)
        self.pt_edges = np.asarray(pt_edges)
        self.incl     = br.Hist2D(pt_edges, y_edges)        # H[pt, y]
        self.per_pdg  = {}                                # pdg -> Hist2D
        self.track    = set(track_pdgs or [])
        self.n_events = 0

    def on_particle_block(self, block, accessor, opts):
        self.n_events += 1
        pairs = accessor.gather_block_arrays(block, ["p0","pz","px","py","pdg"])
        cols  = {k: v for k, v in pairs}
        E,pz,px,py,pdg = cols["p0"], cols["pz"], cols["px"], cols["py"], cols["pdg"]

        # physical mask: avoid invalid rapidity
        m = (E > np.abs(pz))
        if not m.any(): return
        E,pz,px,py,pdg = E[m], pz[m], px[m], py[m], pdg[m]

        y  = 0.5*np.log((E+pz)/(E-pz))
        pt = np.hypot(px, py)

        # inclusive fill
        self.incl.fill(pt, y)

        # optional per-PDG fills
        if self.track:
            # only unique tracked pdgs present in this block
            pdgs_here = np.intersect1d(np.unique(pdg), np.fromiter(self.track, dtype=int))
            for val in pdgs_here:
                sel = (pdg == val)
                H = self.per_pdg.setdefault(int(val), br.Hist2D(self.pt_edges, self.y_edges))
                H.fill(pt, y, mask=sel)

    def merge_from(self, other, opts):
        self.incl.merge_(other.incl)
        for k, H in other.per_pdg.items():
            self.per_pdg.setdefault(k, br.Hist2D(self.pt_edges, self.y_edges))
            self.per_pdg[k].merge_(H)
        self.n_events += getattr(other, "n_events", 0)

    def finalize(self, opts):
        dy  = np.diff(self.y_edges)[0]
        dpt = np.diff(self.pt_edges)[0]
        n_events = max(int(self.n_events), 1)
        norm = n_events * dy * dpt

        self.incl.H /= norm
        for H in self.per_pdg.values():
            H.H /= norm

    def _fmt_val(self,v):
        if isinstance(v, float):
            return f"{round(v, 3):g}"
        return str(v)
    
    def _label_from_keys(self,keys: dict) -> str:
        parts = [f"{k}-{self._fmt_val(keys[k])}" for k in sorted(keys)]
        return "_".join(parts).replace("/", "-")
    
    def save(self, out_dir, keys, opts):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self._label_from_keys(keys)

        path = out_dir / f"dndydpt_{tag}.npz"
        np.savez(
            path,
            H_inclusive=self.incl.H,
            y_edges=self.y_edges,
            pt_edges=self.pt_edges,
            n_events=int(self.n_events),
            **{f"pdg_{k}": v.H for k, v in self.per_pdg.items()},
            keys=keys,
            analysis_name=getattr(self, "name", "dndydpt_python"),
            version=getattr(self, "smash_version", None),
        )

# Register 
edges_y = np.linspace(-4, 4, 31) 
edges_pt = np.linspace(0, 3, 31) 
br.register_python_analysis( "dndydpt_py", 
                            lambda: Dndydpt(edges_y, edges_pt, track_pdgs=[2212, 211, -211]), {} )



