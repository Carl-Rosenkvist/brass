"""Reading SMASH's ASCII thermodynamics output at a point (thermodynamics.dat).

    # SMASH-3.4 thermodynamics output
    # @ point (  0.00,   0.00,   0.00) [fm]
    # net_baryon
    # time [fm], rho_eckart [fm^-3],
    # event 0
     -3.00  0.000000000000
     -2.00  0.000213730618
    # event 1
     ...

    from brass.thermodynamics import read_thermodynamics

    data = read_thermodynamics("thermodynamics.dat")
    data.quantities          # ["rho_eckart"]
    data.events[0].values    # array (times, quantities) of the first event
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ThermodynamicsEvent:
    times: np.ndarray
    values: np.ndarray  # shape (times, quantities)


@dataclass
class ThermodynamicsOutput:
    smash_version: str
    position: tuple[float, float, float] | None
    density_type: str
    quantities: list[str]
    events: list[ThermodynamicsEvent] = field(default_factory=list)


def read_thermodynamics(path: str | Path) -> ThermodynamicsOutput:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header = [line[1:].strip() for line in lines[:4] if line.startswith("#")]
    if len(header) < 4 or "thermodynamics output" not in header[0]:
        raise ValueError(f"{path} is not a SMASH thermodynamics output at a point")

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", header[1])
    position = tuple(float(x) for x in numbers[:3]) if len(numbers) >= 3 else None
    # "time [fm], rho_eckart [fm^-3], ..." -> names without units
    columns = [c.split("[")[0].strip() for c in header[3].split(",") if c.strip()]
    quantities = columns[1:]

    out = ThermodynamicsOutput(header[0].split()[0], position, header[2], quantities)
    rows: list[list[float]] = []

    def close_event():
        if rows:
            data = np.array(rows, dtype=float)
            out.events.append(ThermodynamicsEvent(data[:, 0], data[:, 1:]))
            rows.clear()

    for line in lines[4:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if stripped[1:].split()[:1] == ["event"]:
                close_event()
            continue
        rows.append([float(x) for x in stripped.split()])
    close_event()
    return out
