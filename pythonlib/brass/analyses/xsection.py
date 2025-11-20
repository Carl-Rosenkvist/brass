# xsection.py (updated for new brass API)

import numpy as np
import math
import brass as br

NBINS_DEFAULT = 1000
BMAX_DEFAULT  = 2.5

QUANTITIES = ["t","x","y","z","p0","px","py","pz"]

def two_vecs(cols):
    pA = np.array([cols["p0"][0], cols["px"][0], cols["py"][0], cols["pz"][0]], float)
    pB = np.array([cols["p0"][1], cols["px"][1], cols["py"][1], cols["pz"][1]], float)
    xA = np.array([cols["t"][0],  cols["x"][0],  cols["y"][0],  cols["z"][0]],  float)
    xB = np.array([cols["t"][1],  cols["x"][1],  cols["y"][1],  cols["z"][1]],  float)
    return pA, pB, xA, xB

def transverse_distance(pA, pB, xA, xB):
    p_diff = pA[1:] - pB[1:]
    x_diff = xA[1:] - xB[1:]
    pdp = float(np.dot(p_diff, p_diff))
    xdx = float(np.dot(x_diff, x_diff))
    xdp = float(np.dot(x_diff, p_diff))
    if pdp == 0.0:
        val = xdx
    else:
        val = xdx - (xdp * xdp) / pdp
    return math.sqrt(max(val, 0.0))

def sqrt_s_from(pA, pB):
    ptot = pA + pB
    E = ptot[0]
    p2 = float(np.dot(ptot[1:], ptot[1:]))
    s = E*E - p2
    return math.sqrt(max(s, 0.0))

def energy_index(sqrts, ndigits=5):
    return round(float(sqrts), ndigits)

def calc_tot_xs(bins_scat, bins_tot, bmax, nbins):
    bins_scat = np.asarray(bins_scat, float)
    bins_tot  = np.asarray(bins_tot,  float)

    mask = bins_tot != 0
    if mask.sum() < 2:
        return 0.0, 0.0

    F = bins_scat[mask] / bins_tot[mask]
    if F.size < 2:
        return 0.0, 0.0

    bin_centers = (bmax/nbins) * (np.arange(F.size, dtype=float) + 1.0)
    dF = np.diff(F)

    xs   = -np.sum(bin_centers[:-1]**2 * dF) * math.pi * 10.0
    xs_2 = -np.sum(bin_centers[:-1]**4 * dF) * (math.pi*10.0)**2

    varS = max(0.0, xs_2 - xs*xs)
    return xs, varS


# ================================================================
#                    UPDATED XSECTION ANALYSIS
# ================================================================
class Xsection:
    """
    Updated for new brass API:
      - to_state_dict()
      - finalize(results)
      - no merge_from
      - no save()
    """

    def __init__(self, nbins=NBINS_DEFAULT, bmax=BMAX_DEFAULT):
        self.nbins = int(nbins)
        self.bmax  = float(bmax)

        # by_energy[idx] = { tot, scat, events }
        self.by_energy = {}

        # needed to detect first interaction
        self.first_cols   = None
        self.initial_cols = None

    # ------------------- Brass callbacks -------------------

    def on_particle_block(self, block, accessor, opts):
        if self.initial_cols is None:
            cols = dict(accessor.gather_block_arrays(block))
            if len(cols.get("p0", [])) >= 2:
                self.initial_cols = cols

    def on_interaction_block(self, iblock, accessor, opts):
        if self.first_cols is None:
            cols = dict(accessor.gather_incoming_arrays(iblock))
            if len(cols.get("p0", [])) >= 2:
                self.first_cols = cols

    def _ensure_bin(self, idx):
        if idx not in self.by_energy:
            self.by_energy[idx] = {
                "tot":    np.zeros(self.nbins, dtype=int),
                "scat":   np.zeros(self.nbins, dtype=int),
                "events": 0
            }

    def on_end_block(self, block, accessor, opts):
        cols = self.first_cols if (self.first_cols is not None) else self.initial_cols
        if cols is not None:
            pA, pB, xA, xB = two_vecs(cols)
            R  = transverse_distance(pA, pB, xA, xB)
            sq = sqrt_s_from(pA, pB)
            idx = energy_index(sq)

            self._ensure_bin(idx)
            rec = self.by_energy[idx]
            rec["events"] += 1

            if R <= self.bmax:
                bin_idx = int((R / self.bmax) * self.nbins) - 1
                if bin_idx < 0:  bin_idx = 0
                if bin_idx >= self.nbins: bin_idx = self.nbins - 1

                rec["tot"][bin_idx] += 1
                if self.first_cols is not None:
                    rec["scat"][bin_idx] += 1

        self.first_cols = None
        self.initial_cols = None

    # ------------------- NEW brass API: Export state -------------------

    def to_state_dict(self):
        return {
            "nbins": self.nbins,
            "bmax": self.bmax,
            "by_energy": self.by_energy,
        }

    # ------------------- NEW brass API: Finalize merged results -------------------

    def finalize(self, results):
        """
        results:
           {
             meta_key: {
                "xsection": {nbins, bmax, by_energy}
             }
           }
        We compute xsection rows and store them under:
           results[meta]["xsection_rows"] = [...]
        The pipeline is responsible for writing CSV.
        """
        for meta, analyses in results.items():
            d = analyses.get("xsection")
            if d is None:
                continue

            nbins = d["nbins"]
            bmax  = d["bmax"]
            rows  = []

            for idx, rec in d["by_energy"].items():
                xs, varS = calc_tot_xs(rec["scat"], rec["tot"], bmax, nbins)
                err = math.sqrt(varS) if varS > 0 else 0.0

                rows.append({
                    "sqrt_s": idx,
                    "xsection_mb": xs,
                    "error_mb": err,
                    "variance_mb2": varS,
                    "tot_counts": int(rec["tot"].sum()),
                    "scat_counts": int(rec["scat"].sum()),
                    "events": int(rec["events"]),
                    "nbins": nbins,
                    "bmax": bmax,
                })

            analyses["xsection_rows"] = rows   # pipeline will save these

# Register updated
br.register_python_analysis(
    "xsection",
    lambda: Xsection(NBINS_DEFAULT, BMAX_DEFAULT),
    {},
)
