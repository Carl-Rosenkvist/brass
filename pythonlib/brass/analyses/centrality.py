from pathlib import Path
from typing import Self

import numpy as np

import brass


class Centrality(brass.Analysis):
    """Impact parameter and charged multiplicity per event."""

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)

        self.impact_parameters = []
        self.charged_multiplicities = []

    def data_file_matches(self, path: Path) -> bool:
        return path.name in (
            "particles_binary.bin",
            "particles_custom.bin",
        )

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(
            str(file_path),
            self.quantities,
        )

        n_charged = None

        while (block := reader.read()) is not None:
            if isinstance(block, brass.ParticleBlock):
                charge = block.particles.column("charge")
                n_charged = np.count_nonzero(charge)

            elif isinstance(block, brass.EndBlock):
                if n_charged is None:
                    raise RuntimeError("EndBlock without corresponding ParticleBlock")

                self.impact_parameters.append(block.impact_parameter)
                self.charged_multiplicities.append(n_charged)

                n_charged = None

    def __iadd__(self, other: Self) -> Self:
        self.impact_parameters.extend(other.impact_parameters)
        self.charged_multiplicities.extend(other.charged_multiplicities)

        return self
