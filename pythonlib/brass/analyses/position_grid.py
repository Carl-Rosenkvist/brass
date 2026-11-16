from pathlib import Path
from typing import Self

import numpy as np

import brass

from .multiplicity_vs_time import PARTICLE_FILES


class PositionGrid(brass.Analysis):
    """Particles per cell of an x-y grid at every output time, e.g. the density in a box.

    Needs particle output at several times per event (Only_Final: No) with
    the quantities t, x and y. Positions are binned in --bins x --bins cells
    over 0..L in x and y (L = Modi.Box.Length, or --length), summed over z,
    for all particles or only --pdgs. `density(i)` is the number per event
    and volume, taking the cells as L deep.
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        length = args.length if args.length is not None else self.config.get("Modi", {}).get("Box", {}).get("Length")
        if length is None or length <= 0:
            raise ValueError("give --length, or Modi.Box.Length in the config")
        self.length = float(length)
        self.bins = int(args.bins)
        self.pdgs = list(args.pdgs) if args.pdgs else None
        self.edges = np.linspace(0.0, self.length, self.bins + 1)
        self.times = np.zeros(0)
        self.grids = np.zeros((0, self.bins, self.bins))  # (times, x, y), summed over events
        self.events = 0

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--bins", type=int, default=40, help="Cells along x and y. Default: 40")
        parser.add_argument("--length", type=float, default=None, help="Grid size in fm (default: Modi.Box.Length)")
        parser.add_argument("--pdgs", nargs="+", type=int, default=None, help="Only these PDG codes (default: all)")

    def data_file_matches(self, path: Path) -> bool:
        return path.name in PARTICLE_FILES

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(str(file_path), self.quantities)
        times: list[float] = []
        grids: list[np.ndarray] = []
        current, index, events = None, 0, 0
        while (block := reader.read()) is not None:
            if not isinstance(block, brass.ParticleBlock):
                continue
            event = (block.event_number, block.ensemble_number)
            if event != current:
                current, index = event, 0
                events += 1
            particles = block.particles
            if index == len(times):
                times.append(float(particles.column("t")[0]) if particles.size() else np.nan)
                grids.append(np.zeros((self.bins, self.bins)))
            if particles.size():
                x = np.asarray(particles.column("x"))
                y = np.asarray(particles.column("y"))
                if self.pdgs is not None:
                    keep = np.isin(np.asarray(particles.column("pdg")), self.pdgs)
                    x, y = x[keep], y[keep]
                grids[index] += np.histogram2d(x, y, bins=[self.edges, self.edges])[0]
            index += 1
        self.times = np.array(times)
        self.grids = np.array(grids).reshape(len(times), self.bins, self.bins)
        self.events = events

    def __iadd__(self, other: Self) -> Self:
        if (self.bins, self.length, self.pdgs) != (other.bins, other.length, other.pdgs):
            raise ValueError("cannot add grids with different --bins, --length or --pdgs")
        if len(self.times) != len(other.times) or not np.allclose(self.times, other.times, equal_nan=True):
            raise ValueError("cannot add grids with different output times")
        self.grids = self.grids + other.grids
        self.events += other.events
        return self

    def density(self, time_index: int) -> np.ndarray:
        """Particles per event and fm^3 in each (x, y) cell at one output time."""
        cell_volume = (self.length / self.bins) ** 2 * self.length
        return self.grids[time_index] / max(self.events, 1) / cell_volume
