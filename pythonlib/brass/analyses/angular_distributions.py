import math
from pathlib import Path
from typing import Self

import numpy as np

import brass

from .collision_counts import COLLISION_FILES

Channel = tuple[int, ...]


def _momenta(particles) -> np.ndarray:
    """Four-momenta as rows (p0, px, py, pz)."""
    return np.stack([np.asarray(particles.column(q)) for q in ("p0", "px", "py", "pz")], axis=1)


def _square(p) -> float:
    return float(p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2)


def _p_cm_squared(p1, p2) -> float:
    """Squared centre-of-mass momentum of two particles."""
    s = _square(p1 + p2)
    m1 = math.sqrt(max(_square(p1), 0.0))
    m2 = math.sqrt(max(_square(p2), 0.0))
    return max((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4.0 * s), 0.0)


class AngularDistributions(brass.Analysis):
    """dsigma/dtheta and dsigma/d|t| of 2 -> 2 reactions of one incoming pair.

    For every interaction of the two --incoming particles (in either order)
    the total is counted; for those with two outgoing particles, theta (the
    polar angle of the first outgoing particle, in the frame of the output,
    e.g. the centre of mass) and |t| = |(p_in[0] - p_out[0])^2| are
    histogrammed per channel (sorted outgoing PDGs). The cross section is
    the mean `sigma` of these interactions, so
    dsigma/dtheta = N_bin * sigma / (N_total * dtheta).
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        self.incoming = tuple(sorted(args.incoming))
        self.theta_edges = np.linspace(0.0, math.pi, args.theta_bins + 1)
        self.t_bins = int(args.t_bins)
        self.t_max = args.t_max
        self.t_edges: np.ndarray | None = None
        self.theta_counts: dict[Channel, np.ndarray] = {}
        self.t_counts: dict[Channel, np.ndarray] = {}
        self.total = 0
        self.sigma_sum = 0.0
        self.sigma_n = 0
        self.events = 0

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--incoming", nargs=2, type=int, required=True, metavar=("PDG1", "PDG2"),
                            help="The colliding pair, e.g. 2212 2212.")
        parser.add_argument("--theta-bins", type=int, default=30, help="Bins in theta over [0, pi]. Default: 30")
        parser.add_argument("--t-bins", type=int, default=50, help="Bins in |t| over [0, t_max]. Default: 50")
        parser.add_argument("--t-max", type=float, default=None,
                            help="Upper |t| edge in GeV^2 (default: 4 p_cm^2, the elastic maximum, "
                                 "from the first interaction; |t| above it is not histogrammed).")

    def data_file_matches(self, path: Path) -> bool:
        return path.name in COLLISION_FILES

    def analyze_file(self, file_path):
        thetas: dict[Channel, list[float]] = {}
        ts: dict[Channel, list[float]] = {}
        reader = brass.BinaryReader(str(file_path), self.quantities)
        while (block := reader.read()) is not None:
            if not isinstance(block, brass.InteractionBlock) or block.n_in != 2:
                continue
            if tuple(sorted(int(x) for x in block.incoming.column("pdg"))) != self.incoming:
                continue
            self.total += 1
            if block.n_out != 2:
                continue
            p_in, p_out = _momenta(block.incoming), _momenta(block.outgoing)
            if self.t_max is None:
                self.t_max = 4.0 * _p_cm_squared(p_in[0], p_in[1])
            channel = tuple(sorted(int(x) for x in block.outgoing.column("pdg")))
            p3 = p_out[0]
            theta = math.acos(max(-1.0, min(1.0, p3[3] / math.sqrt(p3[1] ** 2 + p3[2] ** 2 + p3[3] ** 2))))
            thetas.setdefault(channel, []).append(theta)
            ts.setdefault(channel, []).append(abs(_square(p_in[0] - p3)))
            self.sigma_sum += float(block.sigma)
            self.sigma_n += 1

        self.events = reader.end_blocks_read
        self.t_edges = np.linspace(0.0, self.t_max or 1.0, self.t_bins + 1)
        self.theta_counts = {c: np.histogram(v, self.theta_edges)[0].astype(float) for c, v in thetas.items()}
        self.t_counts = {c: np.histogram(v, self.t_edges)[0].astype(float) for c, v in ts.items()}

    def __iadd__(self, other: Self) -> Self:
        if self.incoming != other.incoming or not (
            np.array_equal(self.theta_edges, other.theta_edges)
            and np.allclose(self.t_edges, other.t_edges)
        ):
            raise ValueError("cannot add results with different --incoming or binning")
        for mine, theirs in ((self.theta_counts, other.theta_counts), (self.t_counts, other.t_counts)):
            for channel, counts in theirs.items():
                mine[channel] = mine.get(channel, 0.0) + counts
        self.total += other.total
        self.sigma_sum += other.sigma_sum
        self.sigma_n += other.sigma_n
        self.events += other.events
        return self

    # --- results ----------------------------------------------------------

    @property
    def sigma(self) -> float:
        """Mean total cross section of the 2 -> 2 interactions in mb."""
        return self.sigma_sum / self.sigma_n if self.sigma_n else math.nan

    def channels(self) -> list[Channel]:
        return sorted(self.theta_counts)

    def _sum(self, counts: dict, channels) -> np.ndarray:
        selected = self.channels() if channels is None else [tuple(sorted(c)) for c in channels]
        total = np.zeros(len(next(iter(counts.values()))) if counts else 0)
        for channel in selected:
            if channel in counts:
                total = total + counts[channel]
        return total

    def dsigma_dtheta(self, channels: list[Channel] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(bin centres, dsigma/dtheta in mb/rad), summed over channels (default: all 2 -> 2)."""
        counts = self._sum(self.theta_counts, channels)
        centres = 0.5 * (self.theta_edges[1:] + self.theta_edges[:-1])
        return centres, counts * self.sigma / (max(self.total, 1) * np.diff(self.theta_edges))

    def dsigma_dt(self, channels: list[Channel] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(bin centres, dsigma/d|t| in mb/GeV^2), summed over channels (default: all 2 -> 2)."""
        counts = self._sum(self.t_counts, channels)
        centres = 0.5 * (self.t_edges[1:] + self.t_edges[:-1])
        return centres, counts * self.sigma / (max(self.total, 1) * np.diff(self.t_edges))
