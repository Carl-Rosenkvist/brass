import argparse
from pathlib import Path
from typing import Self

import numpy as np

import brass


class Dileptons(brass.Analysis):
    """Create invariant-mass histograms from dilepton output."""

    def __init__(
        self,
        folder_path: str | Path,
        args: argparse.Namespace,
    ):
        super().__init__(folder_path, args)

        self.edges: dict[str, np.ndarray] = {}
        self.values: dict[str, np.ndarray] = {}

    @classmethod
    def add_arguments(
        cls,
        parser: argparse.ArgumentParser,
    ) -> None:
        parser.add_argument(
            "--bins",
            type=int,
            default=50,
            help="Number of invariant-mass bins",
        )
        parser.add_argument(
            "--mass-min",
            type=float,
            default=0.0,
            help="Lower invariant-mass bound",
        )
        parser.add_argument(
            "--mass-max",
            type=float,
            default=1.5,
            help="Upper invariant-mass bound",
        )
        parser.add_argument(
            "--pair-pdgs",
            type=int,
            nargs=2,
            metavar=("PDG1", "PDG2"),
            default=(-11, 11),
            help="PDG codes of the dilepton pair",
        )
        parser.add_argument(
            "--weight-quantity",
            default="weight",
            help="Quantity used as the histogram weight",
        )

    def data_file_matches(self, path: Path) -> bool:
        return path.name == "Dileptons_custom.bin"

    def analyze_file(self, file_path: str | Path) -> None:
        histogrammer = brass.DileptonHistogrammer(
            quantities=["mass"],
            axes=[
                brass.RegularAxis(
                    self.args.bins,
                    self.args.mass_min,
                    self.args.mass_max,
                )
            ],
            pair_pdgs=tuple(self.args.pair_pdgs),
            weight_quantity=self.args.weight_quantity,
        )
        reader = brass.BinaryReader(
            str(file_path),
            self.quantities,
        )

        histogrammer.fill(reader)
        results = histogrammer.result()

        self.edges.clear()
        self.values.clear()

        for channel, result in results.items():
            self.edges[channel] = np.asarray(
                result.edges[0],
                dtype=float,
            )
            self.values[channel] = np.asarray(
                result.values,
                dtype=float,
            )

    def __iadd__(self, other: Self) -> Self:
        for channel, other_values in other.values.items():
            if channel not in self.values:
                self.edges[channel] = other.edges[channel].copy()
                self.values[channel] = other_values.copy()
                continue

            if not np.array_equal(
                self.edges[channel],
                other.edges[channel],
            ):
                raise ValueError(f"Different bin edges for channel {channel}")

            self.values[channel] += other_values

        self.nevents += other.nevents
        return self
