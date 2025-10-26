import os
import numpy as np
import math
import csv
import sys
import brass as br

NBINS_DEFAULT = 1000
BMAX_DEFAULT  = 2.5
CSV_DEFAULT   = "xsection.csv"

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
    if val < 0.0:
        val = 0.0
    return math.sqrt(val)

def sqrt_s_from(pA, pB):
    ptot = pA + pB
    E = ptot[0]
    p2 = float(np.dot(ptot[1:], ptot[1:]))
    s = E*E - p2
    if s < 0.0:
        s = 0.0
    return math.sqrt(s)

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
    xs_2 = -np.sum(bin_centers[:-1]**4 * dF) * (math.pi * 10.0)**2
    varS = max(0.0, xs_2 - xs**2)
    return xs, varS

class Xsection:
    def __init__(self, nbins=NBINS_DEFAULT, bmax=BMAX_DEFAULT, csv_path_default=CSV_DEFAULT):
        self.nbins = int(nbins)
        self.bmax  = float(bmax)
        self.csv_path_default = str(csv_path_default)
        self.by_energy = {}
        self.n_events = 0
        self.first_cols   = None
        self.initial_cols = None

    def on_particle_block(self, block, accessor, opts):
        if self.initial_cols is None:
            cols = dict(accessor.gather_block_arrays(block, QUANTITIES))
            if len(cols.get("p0", [])) >= 2:
                self.initial_cols = cols

    def on_interaction_block(self, iblock, accessor, opts):
        if self.first_cols is None:
            cols = dict(accessor.gather_incoming_arrays(iblock, QUANTITIES))
            if len(cols.get("p0", [])) >= 2:
                self.first_cols = cols

    def _ensure_energy_bin(self, idx):
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
            idx = energy_index(sq, ndigits=5)
            self._ensure_energy_bin(idx)
            self.by_energy[idx]["events"] += 1
            if R <= self.bmax:
                bin_index = int((R / self.bmax) * self.nbins)
                bin_index = 0 if bin_index == 0 else bin_index - 1
                if bin_index >= self.nbins:
                    bin_index = self.nbins - 1
                self.by_energy[idx]["tot"][bin_index]  += 1
                if self.first_cols is not None:
                    self.by_energy[idx]["scat"][bin_index] += 1
        self.n_events += 1
        self.first_cols = None
        self.initial_cols = None

    def merge_from(self, other, opts):
        if not isinstance(other, Xsection):
            raise TypeError("merge_from expects Xsection")
        if (other.nbins != self.nbins) or (other.bmax != self.bmax):
            raise ValueError(f"Incompatible binning: "
                             f"(nbins,bmax)=({self.nbins},{self.bmax}) vs ({other.nbins},{other.bmax})")
        for idx, rec in other.by_energy.items():
            self._ensure_energy_bin(idx)
            self.by_energy[idx]["tot"]    += rec["tot"]
            self.by_energy[idx]["scat"]   += rec["scat"]
            self.by_energy[idx]["events"] += rec["events"]
        self.n_events += other.n_events
        return self

    def __iadd__(self, other):
        return self.merge_from(other, opts={})

    def _make_rows(self):
        rows = []
        for idx in sorted(self.by_energy.keys()):
            rec = self.by_energy[idx]
            xs, varS = calc_tot_xs(rec["scat"], rec["tot"], self.bmax, self.nbins)
            err = math.sqrt(varS) if varS >= 0.0 else 0.0
            rows.append({
                "sqrt_s": idx,
                "xsection_mb": xs,
                "error_mb": err,
                "variance_mb2": varS,
                "tot_counts": int(rec["tot"].sum()),
                "scat_counts": int(rec["scat"].sum()),
                "events": rec["events"],
                "nbins": self.nbins,
                "bmax": self.bmax,
            })
        return rows

    def save(self, out_dir, keys, opts):
        rows = self._make_rows()
        if not rows:
            print("[xsection] no data to save")
            return
        os.makedirs(out_dir, exist_ok=True)
        csv_name = (opts.get("csv") if isinstance(opts, dict) and "csv" in opts
                    else self.csv_path_default)
        csv_path = os.path.join(out_dir, csv_name)
        fieldnames = ["sqrt_s", "xsection_mb", "error_mb", "variance_mb2",
                      "tot_counts", "scat_counts", "events", "nbins", "bmax"]
        try:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"[xsection] wrote {len(rows)} rows to {csv_path}")
        except Exception as e:
            print(f"[xsection] failed to write CSV: {e}", file=sys.stderr)

br.register_python_analysis(
    "xsection",
    lambda: Xsection(NBINS_DEFAULT, BMAX_DEFAULT, CSV_DEFAULT),
    {},
)
