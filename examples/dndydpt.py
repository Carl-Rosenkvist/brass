# dndydpt_py.py
import numpy as np
import brass as br

class Dndydpt:
    def __init__(self, y_edges, pt_edges, track_pdgs=None):
        self.y_edges  = np.asarray(y_edges)
        self.pt_edges = np.asarray(pt_edges)
        self.Y_BINS   = len(y_edges) - 1
        self.PT_BINS  = len(pt_edges) - 1
        self.H_incl   = np.zeros((self.PT_BINS, self.Y_BINS))
        self.per_pdg  = {}
        self.track    = set(track_pdgs or [])
        self.n_events = 0                      # <-- init

    def on_particle_block(self, block, accessor, opts):
        self.n_events += 1                     # <-- count blocks/events
        pairs = accessor.gather_block_arrays(block, ["p0","pz","px","py","pdg"])
        cols  = {k: v for k, v in pairs}
        E,pz,px,py,pdg = cols["p0"], cols["pz"], cols["px"], cols["py"], cols["pdg"]

        m = (E > np.abs(pz))
        if not m.any(): return
        E,pz,px,py,pdg = E[m], pz[m], px[m], py[m], pdg[m]
        y  = 0.5*np.log((E+pz)/(E-pz))
        pt = np.hypot(px, py)

        by = np.searchsorted(self.y_edges,  y,  side="right") - 1
        bp = np.searchsorted(self.pt_edges, pt, side="right") - 1
        ok = (by>=0)&(by<self.Y_BINS)&(bp>=0)&(bp<self.PT_BINS)
        if not ok.any(): return
        flat = bp[ok]*self.Y_BINS + by[ok]
        self.H_incl.ravel()[:] += np.bincount(flat, minlength=self.H_incl.size)

        if self.track:
            pdg_ok = pdg[ok]
            for val in np.unique(pdg_ok):
                if val not in self.track: continue
                sel = (pdg_ok == val)
                H = self.per_pdg.setdefault(int(val), np.zeros_like(self.H_incl))
                H.ravel()[:] += np.bincount(flat[sel], minlength=H.size)

    def merge_from(self, other, opts):
        self.H_incl += other.H_incl
        for k, H in other.per_pdg.items():
            self.per_pdg.setdefault(k, np.zeros_like(self.H_incl))
            self.per_pdg[k] += H
        # merge counters too
        self.n_events += getattr(other, "n_events", 0)

    def finalize(self, opts):
        dy  = np.diff(self.y_edges)[0]
        dpt = np.diff(self.pt_edges)[0]
        n_events = max(int(self.n_events), 1)  # avoid divide-by-zero
        norm = n_events * dy * dpt
        self.H_incl /= norm
        for H in self.per_pdg.values():
            H /= norm

    def save(self, out_dir, opts):
        np.savez(f"{out_dir}/dndydpt_python.npz",
                 H_inclusive=self.H_incl,
                 y_edges=self.y_edges,
                 pt_edges=self.pt_edges,
                 n_events=int(self.n_events),        # <-- use the counter
                 **{f"pdg_{k}": v for k,v in self.per_pdg.items()})

# Register
edges_y  = np.linspace(-4, 4, 31)
edges_pt = np.linspace(0, 3, 31)
br.register_python_analysis(
    "dndydpt_py",
    lambda: Dndydpt(edges_y, edges_pt, track_pdgs=[2212, 211, -211]),
    {}  # pass empty opts if your binding expects it
)
