from collections import Counter
from pathlib import Path
from typing import Self

import brass

COLLISION_FILES = ("collisions_binary.bin", "collisions_custom.bin")


def interaction_time(block) -> float:
    """Time of an interaction: the time of its first incoming particle."""
    return float(block.incoming.column("t")[0])


class CollisionCounts(brass.Analysis):
    """Number of interactions by process type, after a start time.

    Every interaction block with time > --t-start (the time of its first
    incoming particle) is counted under (process type, n_in, n_out). For
    example, the scattering rate of an elastic box is
    `counts_for(process=1, n_in=2, n_out=2)` per event and per
    (End_Time - t_start).
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        self.t_start = float(args.t_start)
        self.counts: Counter = Counter()
        self.events = 0

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--t-start", type=float, default=float("-inf"),
            help="Count interactions after this time in fm (default: all).",
        )

    def data_file_matches(self, path: Path) -> bool:
        return path.name in COLLISION_FILES

    def analyze_file(self, file_path):
        reader = brass.BinaryReader(str(file_path), self.quantities)
        while (block := reader.read()) is not None:
            if not isinstance(block, brass.InteractionBlock):
                continue
            if block.n_in and interaction_time(block) <= self.t_start:
                continue
            self.counts[(block.process, block.n_in, block.n_out)] += 1
        self.events = reader.end_blocks_read

    def __iadd__(self, other: Self) -> Self:
        if self.t_start != other.t_start:
            raise ValueError("cannot add results with different --t-start")
        self.counts.update(other.counts)
        self.events += other.events
        return self

    def counts_for(self, process: int | None = None, n_in: int | None = None,
                   n_out: int | None = None) -> int:
        """Interactions with the given process type and numbers of particles."""
        return sum(
            count
            for (p, i, o), count in self.counts.items()
            if (process is None or p == process)
            and (n_in is None or i == n_in)
            and (n_out is None or o == n_out)
        )
