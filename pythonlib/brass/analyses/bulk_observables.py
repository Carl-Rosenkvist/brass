import math
from pathlib import Path
from typing import Self

import numpy as np

import brass

# Default binning of the SMASH analysis suite's energy scan.
DEFAULT_PT_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4,
                    1.6, 1.8, 2.0, 2.4, 2.8, 3.2]
DEFAULT_MT0_EDGES = DEFAULT_PT_EDGES

# Outer edges that catch everything outside the requested ranges, so that
# totals are exact. No hadron has |y| or a momentum this large.
FAR = 1.0e6

# name: (axis quantities, weight quantity)
REQUESTS = {
    "y": (("y_rap",), None),
    "pt": (("y_rap", "pt"), None),
    "mt0": (("y_rap", "mt0"), None),
    "sum_pt": (("y_rap",), "pt"),
    "sum_mt0": (("y_rap",), "mt0"),
}


class BulkObservables(brass.Analysis):
    """Multiplicities, midrapidity yields, mean pT and mT - m, and spectra."""

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)

        self.pdgs = list(dict.fromkeys(args.pdgs))
        self.add_pdgs = getattr(args, "add_pdgs", None) or []
        for group in self.add_pdgs:
            if len(group) < 2 or group[0] not in self.pdgs:
                raise ValueError(
                    f"--add-pdgs {group}: give a PDG from --pdgs followed by the PDGs to add to it"
                )
        self.source_pdgs = list(
            dict.fromkeys(self.pdgs + [pdg for group in self.add_pdgs for pdg in group])
        )

        y_min, y_max = args.y_range
        if not y_min < y_max or args.y_bins <= 0:
            raise ValueError("--y-range needs MIN < MAX and --y-bins a positive number")
        if args.midrapidity <= 0:
            raise ValueError("--midrapidity must be positive")
        for name in ("pt_edges", "mt0_edges"):
            edges = getattr(args, name)
            if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])) or edges[0] < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be increasing and >= 0")

        self.y_edges = np.linspace(y_min, y_max, args.y_bins + 1)
        self.midrapidity = float(args.midrapidity)
        self.pt_edges = np.asarray(args.pt_edges, dtype=float)
        self.mt0_edges = np.asarray(args.mt0_edges, dtype=float)
        self.skip_elastic = bool(args.skip_elastic)

        self.histograms: dict[str, dict[int, object]] = {}
        self.events = 0

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--pdgs", nargs="+", type=int, required=True, help="PDG codes to analyze.")
        parser.add_argument(
            "--add-pdgs", nargs="+", type=int, action="append", default=None, metavar="PDG",
            help="Add species: the first PDG receives the others, e.g. --add-pdgs 3122 3212. Repeatable.",
        )
        parser.add_argument("--y-range", nargs=2, type=float, default=[-4.0, 4.0], metavar=("MIN", "MAX"),
                            help="Range of the rapidity spectrum. Default: -4 4")
        parser.add_argument("--y-bins", type=int, default=40, help="Rapidity bins. Default: 40")
        parser.add_argument("--midrapidity", type=float, default=0.5,
                            help="Midrapidity is |y| < this. Default: 0.5")
        parser.add_argument("--pt-edges", nargs="+", type=float, default=DEFAULT_PT_EDGES,
                            help="pT bin edges of the midrapidity spectrum in GeV.")
        parser.add_argument("--mt0-edges", nargs="+", type=float, default=DEFAULT_MT0_EDGES,
                            help="mT - m bin edges of the midrapidity spectrum in GeV.")
        parser.add_argument("--skip-elastic", action="store_true",
                            help="Skip (and do not count) events with exactly two particles.")

    def data_file_matches(self, path: Path) -> bool:
        return path.name in ("particles_binary.bin", "particles_custom.bin")

    def _axis(self, quantity: str):
        m = self.midrapidity
        if quantity == "y_rap":
            return brass.VariableAxis([-FAR, -m, m, FAR])
        edges = self.pt_edges if quantity == "pt" else self.mt0_edges
        return brass.VariableAxis([*edges, FAR])

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(str(file_path), self.quantities, skip_elastic=self.skip_elastic)

        names = ["y_full", *REQUESTS]
        requests = []
        for name in names:
            request = brass.HistogramRequest()
            if name == "y_full":
                request.quantities = ["y_rap"]
                request.axes = [brass.VariableAxis([-FAR, *self.y_edges, FAR])]
            else:
                quantities, weight = REQUESTS[name]
                request.quantities = list(quantities)
                request.axes = [self._axis(q) for q in quantities]
                if weight is not None:
                    request.weight_quantity = weight
            request.group_by = brass.HistogramGroupBy("pdg", self.source_pdgs)
            requests.append(request)

        results = brass.histograms(reader, requests)
        histograms = dict(zip(names, results))

        for group in self.add_pdgs:
            target, *sources = group
            for by_pdg in histograms.values():
                for source in sources:
                    by_pdg[target] += by_pdg[source]

        self.histograms = {
            name: {pdg: by_pdg[pdg] for pdg in self.pdgs} for name, by_pdg in histograms.items()
        }
        self.events = reader.particle_blocks_read

    def __iadd__(self, other: Self) -> Self:
        if self.pdgs != other.pdgs:
            raise ValueError("cannot add results with different --pdgs")
        for name, by_pdg in other.histograms.items():
            for pdg, histogram in by_pdg.items():
                self.histograms[name][pdg] += histogram
        self.events += other.events
        return self

    # --- per-event observables -------------------------------------------

    def _values(self, name: str, pdgs: list[int]) -> np.ndarray:
        total = None
        for pdg in pdgs:
            h = self.histograms[name][pdg]
            values = np.asarray(h.values, dtype=float).reshape(h.shape)
            total = values if total is None else total + values
        return total

    def observables(self, pdgs: int | list[int]) -> dict[str, object]:
        """Per-event observables of one PDG code, or of several added together.

        - total_multiplicity: 4π multiplicity
        - midrapidity_yield: dN/dy for |y| < midrapidity
        - meanpt, meanmt: <pT> and <mT - m> at midrapidity (exact, NaN without particles)
        - yspectra: (y, dN/dy)
        - ptspectra, mtspectra: (bin centres, d²N/(dpT dy) or d²N/(d(mT - m) dy)) at midrapidity
        """
        pdgs = [pdgs] if isinstance(pdgs, int) else list(pdgs)
        n = max(self.events, 1)
        dy = 2.0 * self.midrapidity

        y_full = self._values("y_full", pdgs)
        pt = self._values("pt", pdgs)[1]  # middle rapidity bin
        mt0 = self._values("mt0", pdgs)[1]
        count = pt.sum()
        sum_pt = self._values("sum_pt", pdgs)[1]
        sum_mt0 = self._values("sum_mt0", pdgs)[1]

        def centres(edges):
            return 0.5 * (edges[1:] + edges[:-1])

        return {
            "total_multiplicity": y_full.sum() / n,
            "midrapidity_yield": count / n / dy,
            "meanpt": sum_pt / count if count else math.nan,
            "meanmt": sum_mt0 / count if count else math.nan,
            "yspectra": (centres(self.y_edges), y_full[1:-1] / n / np.diff(self.y_edges)),
            "ptspectra": (centres(self.pt_edges), pt[:-1] / n / np.diff(self.pt_edges) / dy),
            "mtspectra": (centres(self.mt0_edges), mt0[:-1] / n / np.diff(self.mt0_edges) / dy),
        }
