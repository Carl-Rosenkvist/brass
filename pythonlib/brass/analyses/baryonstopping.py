import numpy as np
import brass as br
from brass import HistND


class BaryonStopping:
    def __init__(self, y_edges, abs_pz_edges, track_pdgs=None):
        self.y_edges = np.asarray(y_edges)
        self.abs_pz_edges = np.asarray(abs_pz_edges)

        self.per_pdg: dict[int, dict[str, HistND]] = {}
        self.track = set(track_pdgs or [])
        self.n_events = 0

    def on_interaction_block(self, iblock, accessor, opts):
        pass

    def on_end_block(self, block, accessor, opts):
        pass

    def _make_hists(self):
        return {
            "y": HistND([self.y_edges], track_variance=True),
            "abs_pz": HistND([self.abs_pz_edges], track_variance=True),
            "pt_sum_vs_abs_pz": HistND([self.abs_pz_edges], track_variance=True),
            "pt2_sum_vs_abs_pz": HistND([self.abs_pz_edges], track_variance=True),
        }

    def on_particle_block(self, block, accessor, opts):

        cols = dict(accessor.gather_block_arrays(block))

        E = cols["p0"]
        px = cols["px"]
        py = cols["py"]
        pz = cols["pz"]
        pdg = cols["pdg"]

        self.n_events += 1
        if len(pdg) == 2:
            return
        valid = E > np.abs(pz)
        if not valid.any():
            return

        E = E[valid]
        px = px[valid]
        py = py[valid]
        pz = pz[valid]
        pdg = pdg[valid]

        pt = np.sqrt(px**2 + py**2)
        abs_pz = np.abs(pz)
        y = 0.5 * np.log((E + pz) / (E - pz))

        present_tracked = np.intersect1d(
            np.unique(pdg),
            np.fromiter(self.track, dtype=int),
        )

        for val in present_tracked:
            val = int(val)
            sel = pdg == val

            hists = self.per_pdg.setdefault(val, self._make_hists())

            hists["y"].fill(y, mask=sel)
            hists["abs_pz"].fill(abs_pz, mask=sel)

            hists["pt_sum_vs_abs_pz"].fill(
                abs_pz,
                weights=pt,
                mask=sel,
            )

            hists["pt2_sum_vs_abs_pz"].fill(
                abs_pz,
                weights=pt**2,
                mask=sel,
            )

    def to_state_dict(self):
        return {
            "n_events": int(self.n_events),
            "per_pdg": dict(self.per_pdg),
        }

    def finalize(self, results):
        for meta_key, analyses in results.items():
            d = analyses.get("baryon_stopping")
            if d is None:
                continue

            n_events = max(int(d["n_events"]), 1)
            out = {}

            for pdg_id, hists in d["per_pdg"].items():
                H_y = hists["y"]
                H_abs_pz = hists["abs_pz"]
                H_pt_sum = hists["pt_sum_vs_abs_pz"]
                H_pt2_sum = hists["pt2_sum_vs_abs_pz"]

                y_edges = np.asarray(H_y.edges[0])
                abs_pz_edges = np.asarray(H_abs_pz.edges[0])

                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                abs_pz_centers = 0.5 * (abs_pz_edges[:-1] + abs_pz_edges[1:])

                dy = np.diff(y_edges)
                dpz = np.diff(abs_pz_edges)

                counts_y = np.asarray(H_y.counts, dtype=float)
                counts_abs_pz = np.asarray(H_abs_pz.counts, dtype=float)

                err_counts_y = H_y.errors()
                err_counts_abs_pz = H_abs_pz.errors()

                pt_sum = np.asarray(H_pt_sum.counts, dtype=float)
                pt2_sum = np.asarray(H_pt2_sum.counts, dtype=float)

                dndy = counts_y / (n_events * dy)
                dndy_err = err_counts_y / (n_events * dy)

                # Folded pp distribution:
                # counts contain both +pz and -pz, so divide by 2.
                dndpz = counts_abs_pz / (2.0 * n_events * dpz)
                dndpz_err = err_counts_abs_pz / (2.0 * n_events * dpz)

                mean_pt_vs_pz = np.divide(
                    pt_sum,
                    counts_abs_pz,
                    out=np.full_like(pt_sum, 0.0, dtype=float),
                    where=counts_abs_pz > 0,
                )

                mean_pt2_vs_pz = np.divide(
                    pt2_sum,
                    counts_abs_pz,
                    out=np.full_like(pt2_sum, 0.0, dtype=float),
                    where=counts_abs_pz > 0,
                )

                var_pt_vs_pz = mean_pt2_vs_pz - mean_pt_vs_pz**2
                var_pt_vs_pz = np.maximum(var_pt_vs_pz, 0.0)

                mean_pt_err_vs_pz = np.sqrt(
                    np.divide(
                        var_pt_vs_pz,
                        counts_abs_pz,
                        out=np.full_like(var_pt_vs_pz, np.nan, dtype=float),
                        where=counts_abs_pz > 0,
                    )
                )

                out[int(pdg_id)] = {
                    "y": y_centers,
                    "dndy": dndy,
                    "dndy_err": dndy_err,
                    "pz": abs_pz_centers,
                    "dndpz": dndpz,
                    "dndpz_err": dndpz_err,
                    "mean_pt_vs_pz": mean_pt_vs_pz,
                    "mean_pt_err_vs_pz": mean_pt_err_vs_pz,
                }

            d["per_pdg"] = out

        return results


edges_y = np.linspace(-4.0, 4.0, 31)
edges_abs_pz = np.linspace(0.0, 10.0, 81)

br.register_python_analysis(
    "baryon_stopping",
    lambda: BaryonStopping(
        edges_y,
        edges_abs_pz,
        [
            211,
            -211,
            321,
            -321,
            2212,
            -2212,
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
