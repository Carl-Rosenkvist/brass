from pathlib import Path
from typing import Self

import numpy as np

import brass

from .multiplicity_vs_time import PARTICLE_FILES


class Multiplicities(brass.Analysis):
    """Event-by-event multiplicities and mean pT per PDG code, with errors.

    Every particle block is one event. For each PDG code the sum and the sum
    of squares of the per-event counts are kept, so the mean multiplicity
    has a statistical error; likewise for the per-event mean pT (events
    without the particle do not enter it). With test particles
    (General.Testparticles in the config) the multiplicities are divided by
    their number.
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        self.pdgs = list(dict.fromkeys(args.pdgs))
        n = len(self.pdgs)
        self.testparticles = int(self.config.get("General", {}).get("Testparticles", 1))
        self.events = 0
        self.count_sum = np.zeros(n)
        self.count_squares = np.zeros(n)
        self.pt_events = np.zeros(n)   # events with at least one such particle
        self.pt_sum = np.zeros(n)      # sum of the per-event mean pT
        self.pt_squares = np.zeros(n)

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--pdgs", nargs="+", type=int, required=True, help="PDG codes to count.")

    def data_file_matches(self, path: Path) -> bool:
        return path.name in PARTICLE_FILES

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(str(file_path), self.quantities)
        codes = np.array(self.pdgs)
        while (block := reader.read()) is not None:
            if not isinstance(block, brass.ParticleBlock):
                continue
            self.events += 1
            particles = block.particles
            if particles.size() == 0:
                continue
            pdg = np.asarray(particles.column("pdg"))
            pt = np.hypot(np.asarray(particles.column("px")), np.asarray(particles.column("py")))
            selected = pdg[None, :] == codes[:, None]
            counts = selected.sum(axis=1).astype(float)
            self.count_sum += counts
            self.count_squares += counts**2
            present = counts > 0
            mean_pt = np.where(present, (selected * pt[None, :]).sum(axis=1) / np.maximum(counts, 1), 0.0)
            self.pt_events += present
            self.pt_sum += mean_pt
            self.pt_squares += mean_pt**2

    def __iadd__(self, other: Self) -> Self:
        if self.pdgs != other.pdgs or self.testparticles != other.testparticles:
            raise ValueError("cannot add results with different --pdgs or test particles")
        self.events += other.events
        for name in ("count_sum", "count_squares", "pt_events", "pt_sum", "pt_squares"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    # --- results ----------------------------------------------------------

    @staticmethod
    def _mean_and_error(total, squares, n):
        if n < 1:
            return np.nan, np.nan
        mean = total / n
        if n < 2:
            return mean, 0.0
        variance = max(squares / n - mean**2, 0.0) * n / (n - 1)
        return mean, np.sqrt(variance / n)

    def multiplicity(self, pdg: int) -> tuple[float, float]:
        """Mean number per event (per test particle) and its statistical error."""
        i = self.pdgs.index(pdg)
        mean, error = self._mean_and_error(self.count_sum[i], self.count_squares[i], self.events)
        return mean / self.testparticles, error / self.testparticles

    def mean_pt(self, pdg: int) -> tuple[float, float]:
        """Average over events of the event's mean pT in GeV, and its error."""
        i = self.pdgs.index(pdg)
        return self._mean_and_error(self.pt_sum[i], self.pt_squares[i], int(self.pt_events[i]))
