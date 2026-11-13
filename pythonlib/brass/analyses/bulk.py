from pathlib import Path
from typing import Self

import brass


class Bulk(brass.Analysis):
    """Transverse-mass and rapidity histograms.

    With ``--xf-bins`` it also fills, per PDG, a histogram in Feynman
    x_F = 2 p_z / sqrt(s) and p_T (``xf_histograms``). Its x_F bins are
    exact: the p_z axis spans -sqrt(s)/2 .. sqrt(s)/2 of the run, so x_F
    spectra and <p_T>(x_F) need not be derived from the (y, m_T) bins.
    p_z is taken in the computational frame, which must be the centre of
    mass (collider runs of symmetric systems).
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)

        # Requested PDG codes
        self.pdgs = list(dict.fromkeys(args.pdgs))
        self.add_pdgs = getattr(args, "add_pdgs", None) or []

        # Histogram configuration
        self.y_min, self.y_max = args.y_range
        self.mt_min, self.mt_max = args.mt_range

        self.y_bins = args.y_bins
        self.mt_bins = args.mt_bins

        # Validate histogram ranges
        if self.y_min >= self.y_max:
            raise ValueError("--y-range requires MIN < MAX")

        if self.mt_min < 0 or self.mt_min >= self.mt_max:
            raise ValueError("--mt-range requires 0 <= MIN < MAX")

        if self.y_bins <= 0:
            raise ValueError("--y-bins must be a positive integer")

        if self.mt_bins <= 0:
            raise ValueError("--mt-bins must be a positive integer")

        # Validate histogram combinations
        targets = []

        for group in self.add_pdgs:
            if len(group) < 2:
                raise ValueError(
                    f"Each --add-pdgs group requires at least two PDGs: {group}"
                )

            target, *sources = group

            if target not in self.pdgs:
                raise ValueError(f"Target PDG {target} must be included in --pdgs")

            if len(group) != len(set(group)):
                raise ValueError(f"Duplicate PDGs in --add-pdgs group: {group}")

            targets.append(target)

        if len(targets) != len(set(targets)):
            raise ValueError(
                "Each destination PDG can appear in only one --add-pdgs group"
            )

        # Prevent a source from also being a destination
        source_pdgs = {source for group in self.add_pdgs for source in group[1:]}

        if source_pdgs.intersection(targets):
            raise ValueError(
                "A PDG cannot be both a source and a destination in --add-pdgs groups"
            )

        # All PDGs needed to construct requested histograms
        self.source_pdgs = list(
            dict.fromkeys(self.pdgs + [pdg for group in self.add_pdgs for pdg in group])
        )

        # Optional x_F x p_T histograms
        self.xf_bins = getattr(args, "xf_bins", None)
        self.pt_bins = getattr(args, "pt_bins", 30)
        self.pt_min, self.pt_max = getattr(args, "pt_range", (0.0, 3.0))
        self.sqrts = None
        if self.xf_bins is not None:
            if self.xf_bins <= 0:
                raise ValueError("--xf-bins must be a positive integer")
            if self.pt_bins <= 0:
                raise ValueError("--pt-bins must be a positive integer")
            if self.pt_min < 0 or self.pt_min >= self.pt_max:
                raise ValueError("--pt-range requires 0 <= MIN < MAX")
            self.sqrts = self._collider_sqrts()

        self.histograms = None
        self.xf_histograms = None
        self.events = 0

    def _collider_sqrts(self):
        """sqrt(s_NN) of a collider run in its centre-of-mass frame."""
        collider = self.config.get("Modi", {}).get("Collider", {})
        if "Sqrtsnn" not in collider:
            raise ValueError(
                "--xf-bins needs Modi.Collider.Sqrtsnn in the run's config.yaml"
            )
        frame = collider.get("Calculation_Frame", "center of velocity")
        projectile = collider.get("Projectile", {}).get("Particles")
        target = collider.get("Target", {}).get("Particles")
        if frame == "fixed target" or (
            frame == "center of velocity" and projectile != target
        ):
            raise ValueError(
                "--xf-bins needs the centre-of-mass frame "
                "(Calculation_Frame: center of mass, or a symmetric system)"
            )
        return float(collider["Sqrtsnn"])

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--pdgs",
            nargs="+",
            type=int,
            required=True,
            help="PDG codes to analyze.",
        )

        parser.add_argument(
            "--add-pdgs",
            nargs="+",
            type=int,
            action="append",
            default=None,
            metavar="PDG",
            help=(
                "Combine PDG histograms. The first PDG is the "
                "destination; subsequent PDGs are added to it. "
                "Can be specified multiple times. "
                "Example: --add-pdgs 3122 3212"
            ),
        )

        parser.add_argument(
            "--y-range",
            nargs=2,
            type=float,
            default=[-4.0, 4.0],
            metavar=("MIN", "MAX"),
            help=("Rapidity histogram range. Default: -4.0 4.0"),
        )

        parser.add_argument(
            "--mt-range",
            nargs=2,
            type=float,
            default=[0.0, 10.0],
            metavar=("MIN", "MAX"),
            help=("Transverse-mass histogram range in GeV. Default: 0.0 10.0"),
        )

        parser.add_argument(
            "--y-bins",
            type=int,
            default=50,
            help=("Number of rapidity bins. Default: 50"),
        )

        parser.add_argument(
            "--mt-bins",
            type=int,
            default=100,
            help=("Number of transverse-mass bins. Default: 100"),
        )

        parser.add_argument(
            "--xf-bins",
            type=int,
            default=None,
            help=(
                "Also fill an (x_F, p_T) histogram with this many x_F bins "
                "in -1 < x_F < 1. Off by default."
            ),
        )

        parser.add_argument(
            "--pt-range",
            nargs=2,
            type=float,
            default=[0.0, 3.0],
            metavar=("MIN", "MAX"),
            help="p_T range of the x_F histogram in GeV. Default: 0.0 3.0",
        )

        parser.add_argument(
            "--pt-bins",
            type=int,
            default=30,
            help="Number of p_T bins of the x_F histogram. Default: 30",
        )

    def data_file_matches(self, path: Path) -> bool:
        return path.name in (
            "particles_binary.bin",
            "particles_custom.bin",
        )

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(
            str(file_path),
            self.quantities,
            skip_elastic=True,
        )

        request = brass.HistogramRequest()

        request.quantities = [
            "y_rap",
            "mt",
        ]

        request.axes = [
            brass.RegularAxis(
                self.y_bins,
                self.y_min,
                self.y_max,
            ),
            brass.RegularAxis(
                self.mt_bins,
                self.mt_min,
                self.mt_max,
            ),
        ]

        request.group_by = brass.HistogramGroupBy(
            "pdg",
            self.source_pdgs,
        )

        requests = [request]

        if self.xf_bins is not None:
            xf_request = brass.HistogramRequest()
            xf_request.quantities = ["pz", "pt"]
            half = 0.5 * self.sqrts
            xf_request.axes = [
                brass.RegularAxis(self.xf_bins, -half, half),
                brass.RegularAxis(self.pt_bins, self.pt_min, self.pt_max),
            ]
            xf_request.group_by = brass.HistogramGroupBy("pdg", self.source_pdgs)
            requests.append(xf_request)

        results = brass.histograms(reader, requests)

        self.histograms = self._combine(results[0])
        if self.xf_bins is not None:
            self.xf_histograms = self._combine(results[1])

        self.events = reader.particle_blocks_read

    def _combine(self, histograms):
        """Add --add-pdgs sources to their targets; keep requested PDGs."""
        for group in self.add_pdgs:
            target, *sources = group

            for source in sources:
                histograms[target] += histograms[source]

        return {pdg: histograms[pdg] for pdg in self.pdgs}

    def __iadd__(self, other: Self) -> Self:
        mine = getattr(self, "xf_histograms", None)
        theirs = getattr(other, "xf_histograms", None)
        if (mine is None) != (theirs is None):
            raise ValueError("cannot add Bulk results with and without --xf-bins")
        if mine is not None and self.sqrts != other.sqrts:
            raise ValueError(
                f"cannot add x_F histograms of different energies "
                f"({self.sqrts} and {other.sqrts} GeV)"
            )

        for pdg in other.pdgs:
            if pdg in self.histograms:
                self.histograms[pdg] += other.histograms[pdg]
                if mine is not None:
                    mine[pdg] += theirs[pdg]
            else:
                self.histograms[pdg] = other.histograms[pdg]
                if mine is not None:
                    mine[pdg] = theirs[pdg]

        self.pdgs = list(dict.fromkeys([*self.pdgs, *other.pdgs]))

        self.events += other.events

        return self
