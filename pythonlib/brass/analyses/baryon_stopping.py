from pathlib import Path
from typing import Self

import brass

# Histogram kinds: (axis quantity, weight quantity or None).
KINDS = {
    "xf": ("pz", None),  # particles per x_F bin
    "xf_pt": ("pz", "pt"),  # summed pT per x_F bin
    "y": ("y_rap", None),  # particles per rapidity bin
    "y_mt": ("y_rap", "mt"),  # summed mT per rapidity bin
}


class BaryonStopping(brass.Analysis):
    """Longitudinal spectra in Feynman x_F and rapidity, with mean pT and mT.

    For each PDG, fills one-dimensional histograms of the number of
    particles and of their summed p_T in x_F = 2 p_z/sqrt(s), and of the
    number of particles and their summed m_T in rapidity y. dN/dx_F,
    <p_T>(x_F), dN/dy and <m_T>(y) follow from them exactly, without bins in
    p_T or m_T.

    The x_F bins are exact: the p_z axis spans -sqrt(s)/2 .. sqrt(s)/2 of
    the run (``Modi.Collider.Sqrtsnn``). p_z is taken in the computational
    frame, which must be the centre of mass (collider runs of symmetric
    systems, or ``Calculation_Frame: center of mass``).
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)

        self.pdgs = list(dict.fromkeys(args.pdgs))
        self.add_pdgs = getattr(args, "add_pdgs", None) or []

        self.xf_bins = args.xf_bins
        self.y_min, self.y_max = args.y_range
        self.y_bins = args.y_bins

        if self.xf_bins <= 0:
            raise ValueError("--xf-bins must be a positive integer")
        if self.y_bins <= 0:
            raise ValueError("--y-bins must be a positive integer")
        if self.y_min >= self.y_max:
            raise ValueError("--y-range requires MIN < MAX")

        targets = []
        for group in self.add_pdgs:
            if len(group) < 2:
                raise ValueError(
                    f"Each --add-pdgs group requires at least two PDGs: {group}"
                )
            target = group[0]
            if target not in self.pdgs:
                raise ValueError(f"Target PDG {target} must be included in --pdgs")
            if len(group) != len(set(group)):
                raise ValueError(f"Duplicate PDGs in --add-pdgs group: {group}")
            targets.append(target)
        if len(targets) != len(set(targets)):
            raise ValueError("Each destination PDG can appear in only one --add-pdgs group")
        sources = {pdg for group in self.add_pdgs for pdg in group[1:]}
        if sources.intersection(targets):
            raise ValueError(
                "A PDG cannot be both a source and a destination in --add-pdgs groups"
            )

        self.source_pdgs = list(
            dict.fromkeys(self.pdgs + [pdg for group in self.add_pdgs for pdg in group])
        )

        self.sqrts = self._collider_sqrts()
        self.histograms = None
        self.events = 0

    def _collider_sqrts(self):
        """sqrt(s_NN) of a collider run in its centre-of-mass frame."""
        collider = self.config.get("Modi", {}).get("Collider", {})
        if "Sqrtsnn" not in collider:
            raise ValueError(
                "baryon-stopping needs Modi.Collider.Sqrtsnn in the run's config.yaml"
            )
        frame = collider.get("Calculation_Frame", "center of velocity")
        projectile = collider.get("Projectile", {}).get("Particles")
        target = collider.get("Target", {}).get("Particles")
        if frame == "fixed target" or (
            frame == "center of velocity" and projectile != target
        ):
            raise ValueError(
                "baryon-stopping needs the centre-of-mass frame "
                "(Calculation_Frame: center of mass, or a symmetric system)"
            )
        return float(collider["Sqrtsnn"])

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--pdgs", nargs="+", type=int, required=True, help="PDG codes to analyze."
        )
        parser.add_argument(
            "--add-pdgs",
            nargs="+",
            type=int,
            action="append",
            default=None,
            metavar="PDG",
            help=(
                "Combine PDGs: the first PDG is the destination; the others are "
                "added to it. Can be given several times. Example: --add-pdgs 3122 3212"
            ),
        )
        parser.add_argument(
            "--xf-bins",
            type=int,
            default=80,
            help="Number of x_F bins in -1 < x_F < 1. Default: 80",
        )
        parser.add_argument(
            "--y-range",
            nargs=2,
            type=float,
            default=[-4.0, 4.0],
            metavar=("MIN", "MAX"),
            help="Rapidity range. Default: -4.0 4.0",
        )
        parser.add_argument(
            "--y-bins", type=int, default=40, help="Number of rapidity bins. Default: 40"
        )

    def data_file_matches(self, path: Path) -> bool:
        return path.name in ("particles_binary.bin", "particles_custom.bin")

    def _request(self, kind):
        quantity, weight = KINDS[kind]
        request = brass.HistogramRequest()
        request.quantities = [quantity]
        if quantity == "pz":
            half = 0.5 * self.sqrts
            request.axes = [brass.RegularAxis(self.xf_bins, -half, half)]
        else:
            request.axes = [brass.RegularAxis(self.y_bins, self.y_min, self.y_max)]
        request.group_by = brass.HistogramGroupBy("pdg", self.source_pdgs)
        if weight is not None:
            request.weight_quantity = weight
        return request

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(str(file_path), self.quantities, skip_elastic=True)
        results = brass.histograms(reader, [self._request(kind) for kind in KINDS])

        self.histograms = {}
        for kind, histograms in zip(KINDS, results):
            for group in self.add_pdgs:
                target, *sources = group
                for source in sources:
                    histograms[target] += histograms[source]
            self.histograms[kind] = {pdg: histograms[pdg] for pdg in self.pdgs}

        self.events = reader.particle_blocks_read

    def __iadd__(self, other: Self) -> Self:
        if self.sqrts != other.sqrts:
            raise ValueError(
                f"cannot add results of different energies ({self.sqrts} and "
                f"{other.sqrts} GeV)"
            )
        for kind, histograms in other.histograms.items():
            mine = self.histograms[kind]
            for pdg, histogram in histograms.items():
                if pdg in mine:
                    mine[pdg] += histogram
                else:
                    mine[pdg] = histogram

        self.pdgs = list(dict.fromkeys([*self.pdgs, *other.pdgs]))
        self.events += other.events
        return self
