import math
from pathlib import Path
from typing import Self

import numpy as np

import brass
from brass.haft import ELECTRON, POSITRON, HadesAcceptance, acceptance_files, acceptance_path

Key = tuple[str, int]  # (decay channel, PDG of the decaying particle's mother)

class DileptonSpectra(brass.Analysis):
    """Dilepton mass, pT and rapidity spectra per decay channel, optionally HADES-filtered.

    Every dilepton decay of the dilepton output is one pair, weighted with
    its shining weight. Spectra are kept per decay channel (e.g.
    "113 -> -11 11") and per mother of the decaying particle (pdg_mother1,
    0 if not stored), so that e.g. rho mesons from omega decays can be left
    out. With --acceptance, each pair is passed through the HADES filter as
    in the SMASH analysis suite: both leptons are smeared with the HADES
    resolution, pairs with an opening angle below --opening-angle or a
    lepton momentum outside --lepton-momentum are dropped, and the weight is
    multiplied by the acceptance of both leptons; all kinematics then use the
    smeared momenta. Frames are those of the SMASH output (the lab frame for
    HADES: Calculation_Frame "fixed target").

    Spectra: mass, pt, y (all pairs); mass_below, mass_above (pair momentum
    below / above --pair-momentum-cut); pt_<i>, y_<i> (pairs in mass window
    i of --mass-windows, the last one open).
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        self.mass_edges = np.linspace(0.0, args.mass_max, args.mass_bins + 1)
        self.pt_edges = np.linspace(0.0, args.pt_max, args.pt_bins + 1)
        self.y_edges = np.linspace(-args.y_max, args.y_max, args.y_bins + 1)
        self.windows = list(args.mass_windows)
        if any(b <= a for a, b in zip(self.windows, self.windows[1:])):
            raise ValueError("--mass-windows must be increasing")
        self.pair_momentum_cut = float(args.pair_momentum_cut)
        self.lepton = int(args.lepton)
        self.acceptance_file = None if args.acceptance is None else acceptance_path(args.acceptance)
        self.resolution = int(args.resolution)
        self.opening_angle = float(args.opening_angle)
        self.lepton_momentum = tuple(args.lepton_momentum)
        self.seed = args.seed
        self.spectra: dict[str, dict[Key, np.ndarray]] = {}
        self.events = 0

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--mass-bins", type=int, default=150, help="Default: 150")
        parser.add_argument("--mass-max", type=float, default=2.0, help="Upper mass in GeV. Default: 2")
        parser.add_argument("--pt-bins", type=int, default=80, help="Default: 80")
        parser.add_argument("--pt-max", type=float, default=2.0, help="Upper pT in GeV. Default: 2")
        parser.add_argument("--y-bins", type=int, default=32, help="Default: 32")
        parser.add_argument("--y-max", type=float, default=4.0, help="Rapidity range -Y..Y. Default: 4")
        parser.add_argument("--mass-windows", nargs="+", type=float, default=[0.0, 0.15, 0.47, 0.70],
                            help="Mass window edges in GeV for pt_<i> and y_<i>; the last window is open.")
        parser.add_argument("--pair-momentum-cut", type=float, default=0.8,
                            help="|p| of the pair in GeV for mass_below / mass_above. Default: 0.8")
        parser.add_argument("--lepton", type=int, default=11, help="Lepton PDG (11 or 13). Default: 11")
        parser.add_argument("--acceptance", default=None,
                            help="Apply the HADES filter with this acceptance: a beam time shipped with "
                                 f"brass ({', '.join(acceptance_files())}) or a .acc file")
        parser.add_argument("--resolution", type=int, default=3, choices=(1, 2, 3),
                            help="HADES momentum resolution without resolution tables: 1 low, 2 medium, 3 high")
        parser.add_argument("--opening-angle", type=float, default=9.0,
                            help="With --acceptance: minimal opening angle in degrees. Default: 9")
        parser.add_argument("--lepton-momentum", nargs=2, type=float, default=[0.0, 1000.0],
                            metavar=("MIN", "MAX"), help="With --acceptance: lepton |p| range in GeV")
        parser.add_argument("--seed", type=int, default=None, help="Random seed of the smearing")

    def data_file_matches(self, path: Path) -> bool:
        return path.name.startswith("Dileptons") and path.suffix == ".bin"

    # --- reading -----------------------------------------------------------

    def _pairs(self, file_path):
        """Arrays of all dilepton pairs: channel, mother, p(lepton), p(antilepton), weight."""
        reader = brass.BinaryReader(str(file_path), self.quantities)
        has_mother = "pdg_mother1" in self.quantities
        channels, mothers, minus, plus, weights = [], [], [], [], []
        while (block := reader.read()) is not None:
            if not isinstance(block, brass.InteractionBlock) or block.n_in != 1:
                continue
            out_pdg = np.asarray(block.outgoing.column("pdg"))
            i_minus = np.flatnonzero(out_pdg == self.lepton)
            i_plus = np.flatnonzero(out_pdg == -self.lepton)
            if i_minus.size != 1 or i_plus.size != 1:
                continue
            momenta = np.stack([np.asarray(block.outgoing.column(q)) for q in ("p0", "px", "py", "pz")], axis=1)
            parent = int(block.incoming.column("pdg")[0])
            channels.append(f"{parent} -> {' '.join(str(p) for p in sorted(int(x) for x in out_pdg))}")
            mothers.append(int(block.incoming.column("pdg_mother1")[0]) if has_mother else 0)
            minus.append(momenta[i_minus[0]])
            plus.append(momenta[i_plus[0]])
            weights.append(float(block.sigma))
        self.events = reader.end_blocks_read
        shape = (0, 4)
        return (np.array(channels), np.array(mothers, dtype=np.int64),
                np.array(minus).reshape(-1, 4) if minus else np.zeros(shape),
                np.array(plus).reshape(-1, 4) if plus else np.zeros(shape),
                np.array(weights))

    def _filter(self, minus, plus, weights):
        """HADES smearing, cuts and acceptance (in this order, as in the analysis suite)."""
        filt = HadesAcceptance(self.acceptance_file)
        rng = np.random.default_rng(self.seed)
        minus = filt.smear(minus, ELECTRON, resolution=self.resolution, rng=rng)
        plus = filt.smear(plus, POSITRON, resolution=self.resolution, rng=rng)

        def polar(p4):
            p = np.linalg.norm(p4[:, 1:], axis=1)
            theta = np.degrees(np.arccos(np.clip(p4[:, 3] / np.where(p > 0, p, 1.0), -1, 1)))
            phi = np.degrees(np.arctan2(p4[:, 2], p4[:, 1])) % 360.0
            return p, theta, phi

        p1, th1, ph1 = polar(minus)
        p2, th2, ph2 = polar(plus)
        cos_open = (minus[:, 1:] * plus[:, 1:]).sum(axis=1) / np.where(p1 * p2 > 0, p1 * p2, 1.0)
        opening = np.degrees(np.arccos(np.clip(cos_open, -1, 1)))
        lo, hi = self.lepton_momentum
        keep = (opening >= self.opening_angle) & (p1 >= lo) & (p1 <= hi) & (p2 >= lo) & (p2 <= hi)
        acc = filt.acceptance(ELECTRON, p1, th1, ph1) * filt.acceptance(POSITRON, p2, th2, ph2)
        return minus, plus, weights * np.where(keep, acc, 0.0)

    def analyze_file(self, file_path):
        channels, mothers, minus, plus, weights = self._pairs(file_path)
        if self.acceptance_file is not None and len(weights):
            minus, plus, weights = self._filter(minus, plus, weights)

        pair = minus + plus
        mass = np.sqrt(np.maximum(pair[:, 0] ** 2 - (pair[:, 1:] ** 2).sum(axis=1), 0.0))
        pt = np.hypot(pair[:, 1], pair[:, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            y = 0.5 * np.log((pair[:, 0] + pair[:, 3]) / (pair[:, 0] - pair[:, 3]))
        momentum = np.linalg.norm(pair[:, 1:], axis=1)
        windows = [*self.windows, math.inf]

        selections = {
            "mass": (mass, self.mass_edges, np.ones(len(mass), bool)),
            "pt": (pt, self.pt_edges, np.ones(len(mass), bool)),
            "y": (y, self.y_edges, np.ones(len(mass), bool)),
            "mass_below": (mass, self.mass_edges, momentum < self.pair_momentum_cut),
            "mass_above": (mass, self.mass_edges, momentum > self.pair_momentum_cut),
        }
        for i, (lo, hi) in enumerate(zip(windows, windows[1:])):
            inside = (mass > lo) & (mass < hi)
            selections[f"pt_{i}"] = (pt, self.pt_edges, inside)
            selections[f"y_{i}"] = (y, self.y_edges, inside)

        self.spectra = {name: {} for name in selections}
        keys = sorted(set(zip(channels.tolist(), mothers.tolist())))
        for key in keys:
            mine = (channels == key[0]) & (mothers == key[1]) & (weights > 0)
            for name, (values, edges, selected) in selections.items():
                sel = mine & selected
                if sel.any():
                    self.spectra[name][key] = np.histogram(values[sel], bins=edges, weights=weights[sel])[0]

    def __iadd__(self, other: Self) -> Self:
        for name, by_key in other.spectra.items():
            mine = self.spectra.setdefault(name, {})
            for key, values in by_key.items():
                mine[key] = mine.get(key, 0.0) + values
        self.events += other.events
        return self

    # --- results ----------------------------------------------------------

    def edges(self, spectrum: str) -> np.ndarray:
        if spectrum.startswith("mass"):
            return self.mass_edges
        return self.pt_edges if spectrum.startswith("pt") else self.y_edges

    def channels(self) -> list[str]:
        return sorted({channel for by_key in self.spectra.values() for channel, _ in by_key})

    def mothers(self, channel: str) -> list[int]:
        return sorted({m for by_key in self.spectra.values() for c, m in by_key if c == channel})

    def spectrum(self, spectrum: str = "mass", channels=None, mothers=None,
                 exclude_mothers=()) -> tuple[np.ndarray, np.ndarray]:
        """(bin centres, dN/dx per event) summed over the selected channels and mothers."""
        edges = self.edges(spectrum)
        total = np.zeros(len(edges) - 1)
        for (channel, mother), values in self.spectra.get(spectrum, {}).items():
            if channels is not None and channel not in channels:
                continue
            if mothers is not None and mother not in mothers:
                continue
            if mother in exclude_mothers:
                continue
            total = total + values
        return 0.5 * (edges[1:] + edges[:-1]), total / max(self.events, 1) / np.diff(edges)

    def window_label(self, i: int) -> str:
        windows = [*self.windows, math.inf]
        return f"{windows[i]:g} < M < {windows[i + 1]:g} GeV" if windows[i + 1] < math.inf else f"M > {windows[i]:g} GeV"
