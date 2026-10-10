import argparse
import struct

import numpy as np
import pytest
from brass import AnalysisRunner
from brass.analyses import Bulk
from brass.emulation.bulk import BulkEmulator

PION_PDG = 211
PION_MASS = 0.13957

N_PARTICLES = 10_000


def write_header(file):
    file.write(b"SMSH")
    file.write(struct.pack("<H", 9))
    file.write(struct.pack("<H", 1))

    version = b"SMASH-3.1"
    file.write(struct.pack("<I", len(version)))
    file.write(version)


def write_particles(file, particles):
    if not particles:
        return

    n_fields = len(particles[0])

    if any(len(particle) != n_fields for particle in particles):
        raise ValueError("All particles must have the same number of fields")

    dtype = np.dtype(
        [
            (
                f"f{i}",
                "<i4" if isinstance(value, int) else "<f8",
            )
            for i, value in enumerate(particles[0])
        ]
    )

    array = np.array(
        [tuple(particle) for particle in particles],
        dtype=dtype,
    )

    file.write(array.tobytes())


def write_particle_block(
    file,
    particles,
    event_number=0,
    ensemble_number=0,
):
    file.write(b"p")
    file.write(struct.pack("<i", event_number))
    file.write(struct.pack("<i", ensemble_number))
    file.write(struct.pack("<I", len(particles)))

    write_particles(file, particles)


def generate_particles(
    rng,
    n,
    y_mean,
    y_sigma,
    pt_mean,
):
    y = rng.normal(
        y_mean,
        y_sigma,
        n,
    )

    pt = rng.exponential(
        pt_mean,
        n,
    )

    phi = rng.uniform(
        0.0,
        2.0 * np.pi,
        n,
    )

    mt = np.sqrt(PION_MASS**2 + pt**2)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = mt * np.sinh(y)
    p0 = mt * np.cosh(y)

    return [
        [
            float(energy),
            float(px_i),
            float(py_i),
            float(pz_i),
            PION_PDG,
        ]
        for energy, px_i, py_i, pz_i in zip(
            p0,
            px,
            py,
            pz,
        )
    ]


def make_run(
    folder,
    seed,
    y_mean,
    y_sigma,
    pt_mean,
):
    folder.mkdir()

    (folder / "config.yaml").write_text(
        f"""
Meta:
  y_mean: {y_mean}
  y_sigma: {y_sigma}
  pt_mean: {pt_mean}

Output:
  Particles:
    Quantities:
      - p0
      - px
      - py
      - pz
      - pdg
"""
    )

    rng = np.random.default_rng(seed)

    particles = generate_particles(
        rng,
        N_PARTICLES,
        y_mean,
        y_sigma,
        pt_mean,
    )

    path = folder / "particles_binary.bin"

    with path.open("wb") as file:
        write_header(file)
        write_particle_block(
            file,
            particles,
        )


def test_bulk_emulator(tmp_path):
    parameters = [
        (-0.5, 0.8, 0.3),
        (-0.5, 0.8, 0.5),
        (-0.5, 1.2, 0.3),
        (-0.5, 1.2, 0.5),
        (0.5, 0.8, 0.3),
        (0.5, 0.8, 0.5),
        (0.5, 1.2, 0.3),
        (0.5, 1.2, 0.5),
    ]

    runs = []

    for i, parameter in enumerate(parameters):
        run = tmp_path / f"run{i}"

        make_run(
            run,
            seed=42 + i,
            y_mean=parameter[0],
            y_sigma=parameter[1],
            pt_mean=parameter[2],
        )

        runs.append(run)

    runner = AnalysisRunner(
        Bulk,
        runs,
        args=argparse.Namespace(
            pdgs=[PION_PDG],
        ),
        max_workers=2,
    )

    results = runner.run()

    emulator = BulkEmulator(
        data=results,
        parameter_paths={
            "y_mean": ("Meta", "y_mean"),
            "y_sigma": ("Meta", "y_sigma"),
            "pt_mean": ("Meta", "pt_mean"),
        },
        n_components=0.95,
    ).fit()

    target = np.array(
        [0.0, 1.0, 0.4],
    )

    hist = emulator.predict(
        target,
        PION_PDG,
    )[0]

    obs = emulator.observables(
        PION_PDG,
    )

    weights = hist * obs.dy[:, None] * obs.dmt[None, :]

    total = np.sum(weights)

    y_mean = np.sum(obs.y[:, None] * weights) / total

    y_sigma = np.sqrt(np.sum((obs.y[:, None] - y_mean) ** 2 * weights) / total)

    pt = np.sqrt(
        np.maximum(
            obs.mt**2 - PION_MASS**2,
            0.0,
        )
    )

    pt_mean = np.sum(pt[None, :] * weights) / total

    assert y_mean == pytest.approx(
        target[0],
        abs=0.01,
    )

    assert y_sigma == pytest.approx(
        target[1],
        abs=0.2,
    )

    assert pt_mean == pytest.approx(
        target[2],
        abs=0.05,
    )
