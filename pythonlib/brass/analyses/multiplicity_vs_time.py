from pathlib import Path
from typing import Self

import numpy as np

import brass

PARTICLE_FILES = ("particles_binary.bin", "particles_custom.bin")


class MultiplicityVsTime(brass.Analysis):
    """Mean multiplicity of each PDG code at every output time.

    Needs particle output at several times per event (Only_Final: No) and
    the quantity t. For each output time and PDG code, the sum and the sum
    of squares of the per-event counts are kept, so that the mean and its
    statistical error follow (`mean`, `error`). Counts include test
    particles.
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        self.pdgs = list(dict.fromkeys(args.pdgs))
        self.times = np.zeros(0)
        self.sums = np.zeros((len(self.pdgs), 0))
        self.squares = np.zeros((len(self.pdgs), 0))
        self.events = 0

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--pdgs", nargs="+", type=int, required=True, help="PDG codes to count.")

    def data_file_matches(self, path: Path) -> bool:
        return path.name in PARTICLE_FILES

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(str(file_path), self.quantities)
        codes = np.array(self.pdgs)
        times: list[float] = []
        sums: list[np.ndarray] = []
        squares: list[np.ndarray] = []
        current = None
        index = 0
        events = 0
        while (block := reader.read()) is not None:
            if not isinstance(block, brass.ParticleBlock):
                continue
            event = (block.event_number, block.ensemble_number)
            if event != current:
                current, index = event, 0
                events += 1
            particles = block.particles
            pdg = np.asarray(particles.column("pdg"))
            counts = (pdg[None, :] == codes[:, None]).sum(axis=1).astype(float)
            if index == len(times):
                time = float(particles.column("t")[0]) if particles.size() else np.nan
                times.append(time)
                sums.append(np.zeros(len(codes)))
                squares.append(np.zeros(len(codes)))
            sums[index] += counts
            squares[index] += counts**2
            index += 1

        self.times = np.array(times)
        self.sums = np.array(sums).T.reshape(len(codes), -1)
        self.squares = np.array(squares).T.reshape(len(codes), -1)
        self.events = events

    def __iadd__(self, other: Self) -> Self:
        if self.pdgs != other.pdgs:
            raise ValueError("cannot add results with different --pdgs")
        if len(self.times) != len(other.times) or not np.allclose(self.times, other.times, equal_nan=True):
            raise ValueError("cannot add results with different output times")
        self.sums = self.sums + other.sums
        self.squares = self.squares + other.squares
        self.events += other.events
        return self

    def mean(self, pdg: int) -> np.ndarray:
        """Mean count per event at each output time."""
        return self.sums[self.pdgs.index(pdg)] / max(self.events, 1)

    def error(self, pdg: int) -> np.ndarray:
        """Statistical error of the mean: sqrt(variance / (events - 1))."""
        i, n = self.pdgs.index(pdg), self.events
        if n < 2:
            return np.zeros(len(self.times))
        mean = self.sums[i] / n
        variance = np.maximum(self.squares[i] / n - mean**2, 0.0)
        return np.sqrt(variance / (n - 1))
