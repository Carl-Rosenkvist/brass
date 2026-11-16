from pathlib import Path
from typing import Self

import numpy as np

import brass
from brass.thermodynamics import read_thermodynamics


class Thermodynamics(brass.Analysis):
    """Event average of SMASH's thermodynamics output at a point, against time.

    Reads thermodynamics.dat (Output.Thermodynamics with Format ASCII and a
    Position) and keeps, per output time and quantity, the sum and the sum
    of squares over events, for the mean and its statistical error. For
    example the Eckart baryon density at the centre of a collision.
    """

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)
        self.names: list[str] = []
        self.density_type = ""
        self.position = None
        self.times = np.zeros(0)
        self.sums = np.zeros((0, 0))     # (times, quantities)
        self.squares = np.zeros((0, 0))
        self.events = 0

    def extract_output_quantities(self) -> list[str]:
        # an ASCII output: the quantities are named in the file itself
        return []

    def data_file_matches(self, path: Path) -> bool:
        return path.name == "thermodynamics.dat"

    def analyze_file(self, file_path):
        data = read_thermodynamics(file_path)
        self.names = data.quantities
        self.density_type = data.density_type
        self.position = data.position
        if not data.events:
            return
        self.times = data.events[0].times
        self.sums = np.zeros_like(data.events[0].values)
        self.squares = np.zeros_like(self.sums)
        for event in data.events:
            if len(event.times) != len(self.times) or not np.allclose(event.times, self.times):
                raise ValueError(f"{file_path}: events have different output times")
            self.sums += event.values
            self.squares += event.values**2
        self.events = len(data.events)

    def __iadd__(self, other: Self) -> Self:
        if other.events == 0:
            return self
        if self.events == 0:
            self.__dict__.update(other.__dict__)
            return self
        if self.names != other.names or len(self.times) != len(other.times) or not np.allclose(self.times, other.times):
            raise ValueError("cannot add thermodynamics results with different quantities or output times")
        self.sums = self.sums + other.sums
        self.squares = self.squares + other.squares
        self.events += other.events
        return self

    def mean(self, quantity: str) -> np.ndarray:
        """Event average at each output time."""
        return self.sums[:, self.names.index(quantity)] / max(self.events, 1)

    def error(self, quantity: str) -> np.ndarray:
        """Statistical error of the mean."""
        n = self.events
        if n < 2:
            return np.zeros(len(self.times))
        i = self.names.index(quantity)
        mean = self.sums[:, i] / n
        variance = np.maximum(self.squares[:, i] / n - mean**2, 0.0) * n / (n - 1)
        return np.sqrt(variance / n)
