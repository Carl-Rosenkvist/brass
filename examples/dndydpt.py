import numpy as np
import brass as br
from pathlib import Path


class Dndydpt:
    def __init__(self, y_edges, pt_edges, track_pdgs=None):
        self.y_edges = np.asarray(y_edges)
        self.pt_edges = np.asarray(pt_edges)

        # 2D histogram over (pt, y), with variance tracking for errors
        self.incl = br.HistND([self.pt_edges, self.y_edges], track_variance=True)
        self.per_pdg = {}  # pdg -> HistND([pt, y], track_variance=True)
        self.track = set(track_pdgs or [])
        self.n_events = 0

    def on_particle_block(self, block, accessor, opts):
        self.n_events += 1
        pairs = accessor.gather_block_arrays(block, ["p0", "pz", "px", "py", "pdg"])
        cols = {k: v for k, v in pairs}
        E, pz, px, py, pdg = cols["p0"], cols["pz"], cols["px"], cols["py"], cols["pdg"]

        m = E > np.abs(pz)  # avoid invalid rapidity
        if not m.any():
            return
        E, pz, px, py, pdg = E[m], pz[m], px[m], py[m], pdg[m]

        y = 0.5 * np.log((E + pz) / (E - pz))
        pt = np.hypot(px, py)

        # inclusive fill
        self.incl.fill(pt, y)

        # optional per-PDG fills
        if self.track:
            pdgs_here = np.intersect1d(
                np.unique(pdg), np.fromiter(self.track, dtype=int)
            )
            for val in pdgs_here:
                sel = pdg == val
                H = self.per_pdg.setdefault(
                    int(val),
                    br.HistND([self.pt_edges, self.y_edges], track_variance=True),
                )
                H.fill(pt, y, mask=sel)

    def merge_from(self, other, opts):
        self.incl.merge_(other.incl)
        for k, H in other.per_pdg.items():
            self.per_pdg.setdefault(
                k, br.HistND([self.pt_edges, self.y_edges], track_variance=True)
            )
            self.per_pdg[k].merge_(H)
        self.n_events += getattr(other, "n_events", 0)

    def finalize(self, opts):
        """
        Normalize to d^2N/(dy dpt) per event, and compute per-bin errors.
        Error per bin = sqrt(variance), with variance propagated under normalization.
        """
        dy = np.diff(self.y_edges)  # (ny,)
        dpt = np.diff(self.pt_edges)  # (npt,)
        area = dpt[:, None] * dy[None, :]  # (npt, ny), supports non-uniform bins
        n_ev = max(int(self.n_events), 1)

        # Normalize counts -> per-event per (dy dpt)
        self._normalize_hist2d_(self.incl, n_ev, area)
        for H in self.per_pdg.values():
            self._normalize_hist2d_(H, n_ev, area)

    @staticmethod
    def _normalize_hist2d_(H, n_ev, area):
        """
        Counts and variance normalization:
          value_norm = counts / (n_ev * area)
          var_norm   = sumw2 / (n_ev^2 * area^2)   (since errors() = sqrt(sumw2))
        """
        H.counts[:] = H.counts / (n_ev * area)
        if getattr(H, "sumw2", None) is not None:
            H.sumw2[:] = H.sumw2 / ((n_ev * area) ** 2)

    def _fmt_val(self, v):
        return f"{round(v, 3):g}" if isinstance(v, float) else str(v)

    def _label_from_keys(self, keys: dict) -> str:
        parts = [f"{k}-{self._fmt_val(keys[k])}" for k in sorted(keys)]
        return "_".join(parts).replace("/", "-")

    def save(self, out_dir, keys, opts):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self._label_from_keys(keys)
        path = out_dir / f"dndydpt_{tag}.npz"

        # Build dict of per-PDG outputs: values + errors (if available)
        pdg_arrays = {}
        for k, H in self.per_pdg.items():
            pdg_arrays[f"pdg_{k}"] = H.counts
            if getattr(H, "sumw2", None) is not None:
                pdg_arrays[f"pdg_{k}_err"] = np.sqrt(H.sumw2)

        # Inclusive arrays: values + errors
        extras = {}
        if getattr(self.incl, "sumw2", None) is not None:
            extras["H_inclusive_err"] = np.sqrt(self.incl.sumw2)

        np.savez(
            path,
            H_inclusive=self.incl.counts,
            y_edges=self.y_edges,
            pt_edges=self.pt_edges,
            n_events=int(self.n_events),
            **pdg_arrays,
            **extras,
            keys=keys,
            analysis_name=getattr(self, "name", "dndydpt_python"),
            version=getattr(self, "smash_version", None),
        )


# Register
edges_y = np.linspace(-4, 4, 31)
edges_pt = np.linspace(0.0, 3.0, 31)

br.register_python_analysis(
    "dndydpt_py",
    lambda: Dndydpt(edges_y, edges_pt, track_pdgs=[2212, 211, -211]),
    {},
)
