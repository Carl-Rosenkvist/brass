from pathlib import Path
from typing import Self

import brass


class Bulk(brass.Analysis):
    """Transverse momentum and rapidity histogram."""

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)

        self.pdgs = list(dict.fromkeys(args.pdgs))
        self.add_pdgs = args.add_pdgs or []

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

        # Prevent a source from also being a destination.
        # This avoids order-dependent combinations.
        source_pdgs = {source for group in self.add_pdgs for source in group[1:]}

        if source_pdgs.intersection(targets):
            raise ValueError(
                "A PDG cannot be both a source and a destination in --add-pdgs groups"
            )

        # All PDGs needed to construct the requested histograms
        self.source_pdgs = list(
            dict.fromkeys(self.pdgs + [pdg for group in self.add_pdgs for pdg in group])
        )

        self.histograms = None
        self.events = 0

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
        request.quantities = ["y_rap", "mt"]
        request.axes = [
            brass.RegularAxis(50, -4.0, 4.0),
            brass.RegularAxis(50, 0.0, 2.0),
        ]
        request.group_by = brass.HistogramGroupBy(
            "pdg",
            self.source_pdgs,
        )

        histograms = brass.histogram(
            reader,
            request,
        )

        # Add source histograms to their destination histograms
        for group in self.add_pdgs:
            target, *sources = group

            for source in sources:
                histograms[target] += histograms[source]

        # Only retain the PDGs explicitly requested for output
        self.histograms = {pdg: histograms[pdg] for pdg in self.pdgs}

        self.events = reader.particle_blocks_read

    def __iadd__(self, other: Self) -> Self:
        for pdg in other.pdgs:
            if pdg in self.histograms:
                self.histograms[pdg] += other.histograms[pdg]
            else:
                self.histograms[pdg] = other.histograms[pdg]

        self.pdgs = list(dict.fromkeys([*self.pdgs, *other.pdgs]))

        self.events += other.events

        return self
