from pathlib import Path
from typing import Self

import brass


class Bulk(brass.Analysis):
    """Transverse momentum and rapidity histogram"""

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)

        self.pdgs = args.pdgs
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

    def data_file_matches(self, path: Path) -> bool:
        return (
            path.name == "particles_binary.bin" or path.name == "particles_custom.bin"
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
            self.pdgs,
        )

        self.histograms = brass.histogram(
            reader,
            request,
        )

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
