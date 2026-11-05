"""Write synthetic SMASH binary output.

Useful for tests, benchmarks and examples that need SMASH files without
running SMASH::

    from brass.testing import SmashBinaryWriter

    with SmashBinaryWriter("particles_binary.bin", ["p0", "px", "py", "pz", "pdg"]) as w:
        w.particles([[1.0, 0.1, 0.2, 0.3, 211]], event=0)
        w.end(event=0, impact_parameter=3.2)

Particles can be given as rows (a sequence of sequences in quantity order),
as a mapping from quantity name to column, or as a structured NumPy array.
"""

import argparse
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Quantities SMASH stores as int32; all others are float64.
# Must match quantity_type() in include/quantities.h.
INT32_QUANTITIES = frozenset(
    {
        "pdg",
        "id",
        "charge",
        "ncoll",
        "proc_id_origin",
        "proc_type_origin",
        "pdg_mother1",
        "pdg_mother2",
        "baryon_number",
        "strangeness",
    }
)

# File each analysis looks for, by Output section.
DEFAULT_FILENAMES = {
    "Particles": "particles_binary.bin",
    "Dileptons": "Dileptons_custom.bin",
    "Collisions": "collisions_binary.bin",
}


def particle_dtype(quantities) -> np.dtype:
    """Packed little-endian record type for one particle."""
    return np.dtype(
        [(q, "<i4" if q in INT32_QUANTITIES else "<f8") for q in quantities]
    )


class SmashBinaryWriter:
    """Write a SMASH binary file block by block."""

    def __init__(
        self,
        path,
        quantities,
        *,
        smash_version: str = "SMASH-3.1",
        format_version: int = 9,
        format_variant: int = 1,
    ):
        self.path = Path(path)
        self.quantities = list(quantities)
        self.dtype = particle_dtype(self.quantities)

        self._file = self.path.open("wb")

        version = smash_version.encode()
        self._file.write(b"SMSH")
        self._file.write(
            struct.pack("<HHI", format_version, format_variant, len(version))
        )
        self._file.write(version)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._file.close()

    def particles(self, particles, *, event: int = 0, ensemble: int = 0) -> None:
        """Write a particle block ('p')."""
        data = self.pack(particles)

        self._file.write(b"p")
        self._file.write(struct.pack("<iiI", event, ensemble, len(data)))
        self._file.write(data.tobytes())

    def interaction(
        self,
        incoming,
        outgoing,
        *,
        rho: float = 0.0,
        sigma: float = 0.0,
        sigma_p: float = 0.0,
        process: int = 0,
    ) -> None:
        """Write an interaction block ('i')."""
        incoming = self.pack(incoming)
        outgoing = self.pack(outgoing)

        self._file.write(b"i")
        self._file.write(struct.pack("<ii", len(incoming), len(outgoing)))
        self._file.write(struct.pack("<dddi", rho, sigma, sigma_p, process))
        self._file.write(incoming.tobytes())
        self._file.write(outgoing.tobytes())

    def end(
        self,
        *,
        event: int = 0,
        ensemble: int = 0,
        impact_parameter: float = 0.0,
        empty: bool = False,
    ) -> None:
        """Write an end-of-event block ('f')."""
        self._file.write(b"f")
        self._file.write(
            struct.pack("<Iid?", event, ensemble, impact_parameter, empty)
        )

    def pack(self, particles) -> np.ndarray:
        """Convert rows, a column mapping or a structured array to records."""
        if isinstance(particles, np.ndarray) and particles.dtype.names:
            columns = [particles[q] for q in self.quantities]
        elif isinstance(particles, Mapping):
            missing = [q for q in self.quantities if q not in particles]
            if missing:
                raise KeyError(f"missing particle quantities: {missing}")
            columns = [np.asarray(particles[q]) for q in self.quantities]
        else:
            rows = list(particles)
            n_fields = len(self.quantities)
            if any(len(row) != n_fields for row in rows):
                raise ValueError(
                    f"every particle row needs {n_fields} values "
                    f"({self.quantities})"
                )
            columns = list(zip(*rows)) if rows else [()] * n_fields

        n = len(columns[0]) if columns else 0
        out = np.empty(n, dtype=self.dtype)

        for q, column in zip(self.quantities, columns):
            out[q] = column

        return out


def write_run(
    folder,
    quantities,
    *,
    output: str = "Particles",
    config: dict[str, Any] | None = None,
    filename: str | None = None,
    **writer_kwargs,
) -> SmashBinaryWriter:
    """Create a SMASH-like run folder and return a writer for its data file.

    Writes ``config.yaml`` with ``Output.<output>.Quantities`` set, merged
    into ``config``, so the folder can be read by a ``brass.Analysis``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    full_config = dict(config or {})
    full_config["Output"] = {
        **full_config.get("Output", {}),
        output: {"Quantities": list(quantities)},
    }

    with (folder / "config.yaml").open("w") as file:
        yaml.safe_dump(full_config, file, sort_keys=False)

    if filename is None:
        filename = DEFAULT_FILENAMES[output]

    return SmashBinaryWriter(folder / filename, quantities, **writer_kwargs)


def generate_particles(
    rng: np.random.Generator,
    n: int,
    *,
    pdg: int,
    mass: float,
    y_mean: float = 0.0,
    y_sigma: float = 1.0,
    pt_mean: float = 0.4,
) -> dict[str, np.ndarray]:
    """Particles with Gaussian rapidity and exponential transverse momentum.

    Returns columns for every momentum quantity plus the generated ``y``,
    ``pt`` and ``mt``, so tests can compare against the truth. Extra columns
    are ignored when writing.
    """
    y = rng.normal(y_mean, y_sigma, n)
    pt = rng.exponential(pt_mean, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)

    mt = np.sqrt(mass**2 + pt**2)

    return {
        "p0": mt * np.cosh(y),
        "px": pt * np.cos(phi),
        "py": pt * np.sin(phi),
        "pz": mt * np.sinh(y),
        "mass": np.full(n, mass),
        "pdg": np.full(n, pdg, dtype=np.int32),
        "y": y,
        "pt": pt,
        "mt": mt,
    }


def concatenate_particles(*parts: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Join column mappings that share the same keys."""
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts]) for key in keys}


def analysis_args(analysis_type, *argv: str) -> argparse.Namespace:
    """Parse arguments for an analysis with its real CLI defaults."""
    parser = argparse.ArgumentParser()
    analysis_type.add_arguments(parser)
    return parser.parse_args(list(argv))
