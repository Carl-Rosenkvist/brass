from pathlib import Path
from typing import Self

import numpy as np

import brass

from .collision_counts import COLLISION_FILES, interaction_time

# Default binning per variable: (min, max, bins)
DEFAULT_BINNING = {"sqrts": (0.0, 5.0, 100), "t": (0.0, 30.0, 60)}

Reaction = tuple[tuple[int, ...], tuple[int, ...]]


def _four_momentum(particles) -> np.ndarray:
    return np.stack([np.asarray(particles.column(q)) for q in ("p0", "px", "py", "pz")])


def _square(p: np.ndarray) -> float:
    return float(p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2)


class ReactionCounts(brass.Analysis):
    """Interactions per reaction, binned in sqrt(s) or |t|, for detailed balance.

    A reaction is (sorted incoming PDGs, sorted outgoing PDGs), so a decay is
    the inverse of the corresponding 2 -> 1 formation. Interactions with time
    > --t-start are counted; 1 -> 1 interactions (wall crossings) are
    skipped. With --variable t only 2 -> 2 reactions are counted, binned in
    |t| = |(p_in[0] - p_out[0])^2|; with sqrts (default) all reactions are
    binned in the invariant mass of the incoming particles.
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        self.variable = args.variable
        lo, hi, bins = DEFAULT_BINNING[self.variable]
        if args.range is not None:
            lo, hi = args.range
        if args.bins is not None:
            bins = args.bins
        if not lo < hi or bins <= 0:
            raise ValueError("--range needs MIN < MAX and --bins a positive number")
        self.edges = np.linspace(lo, hi, bins + 1)
        self.t_start = float(args.t_start)
        self.end_time = float(self.config.get("General", {}).get("End_Time", np.nan))
        self.counts: dict[Reaction, np.ndarray] = {}
        self.events = 0

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--t-start", type=float, default=float("-inf"),
                            help="Count interactions after this time in fm, e.g. "
                                 "once the box is in equilibrium (default: all).")
        parser.add_argument("--variable", choices=sorted(DEFAULT_BINNING), default="sqrts",
                            help="Bin in sqrt(s) of the incoming particles (default) or in |t| (2 -> 2 only).")
        parser.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"),
                            help="Binning range (default: 0 5 GeV for sqrts, 0 30 GeV^2 for t).")
        parser.add_argument("--bins", type=int, help="Number of bins (default: 100 for sqrts, 60 for t).")

    def data_file_matches(self, path: Path) -> bool:
        return path.name in COLLISION_FILES

    def analyze_file(self, file_path):
        values: dict[Reaction, list[float]] = {}
        reader = brass.BinaryReader(str(file_path), self.quantities)
        while (block := reader.read()) is not None:
            if not isinstance(block, brass.InteractionBlock):
                continue
            if block.n_in == 1 and block.n_out == 1:
                continue
            if self.variable == "t" and not (block.n_in == 2 and block.n_out == 2):
                continue
            if interaction_time(block) <= self.t_start:
                continue
            incoming, outgoing = block.incoming, block.outgoing
            p_in = _four_momentum(incoming)
            if self.variable == "sqrts":
                value = np.sqrt(max(_square(p_in.sum(axis=1)), 0.0))
            else:
                value = abs(_square(p_in[:, 0] - _four_momentum(outgoing)[:, 0]))
            key = (
                tuple(sorted(int(x) for x in incoming.column("pdg"))),
                tuple(sorted(int(x) for x in outgoing.column("pdg"))),
            )
            values.setdefault(key, []).append(value)

        self.counts = {
            key: np.histogram(vals, bins=self.edges)[0].astype(float)
            for key, vals in values.items()
        }
        self.events = reader.end_blocks_read

    def __iadd__(self, other: Self) -> Self:
        if (self.variable, self.t_start) != (other.variable, other.t_start) or not np.array_equal(
            self.edges, other.edges
        ):
            raise ValueError("cannot add results with different --variable, --t-start or binning")
        for key, counts in other.counts.items():
            self.counts[key] = self.counts.get(key, 0.0) + counts
        self.events += other.events
        return self

    # --- results ----------------------------------------------------------

    @staticmethod
    def inverse(reaction: Reaction) -> Reaction:
        return (reaction[1], reaction[0])

    def histogram(self, reaction: Reaction) -> np.ndarray:
        """Counts per bin (zeros if the reaction never happened)."""
        return self.counts.get(reaction, np.zeros(len(self.edges) - 1))

    def pairs(self) -> list[tuple[Reaction, Reaction]]:
        """Each reaction with its inverse, once per pair (forward first)."""
        seen, pairs = set(), []
        for reaction in sorted(self.counts):
            if reaction in seen:
                continue
            backward = self.inverse(reaction)
            seen.update({reaction, backward})
            pairs.append((reaction, backward))
        return pairs

    def rate(self, reaction: Reaction) -> np.ndarray:
        """Reactions per event, per fm of time after t_start and per bin width."""
        duration = self.end_time - self.t_start
        return self.histogram(reaction) / (max(self.events, 1) * duration * np.diff(self.edges))

    def most_violating(self, n: int = 6) -> list[tuple[Reaction, Reaction]]:
        """The n pairs with the largest |sqrt(N_forward) - sqrt(N_backward)|."""
        def deviation(pair):
            forward, backward = (self.histogram(r).sum() for r in pair)
            return abs(np.sqrt(forward) - np.sqrt(backward))
        return sorted(self.pairs(), key=deviation, reverse=True)[:n]
