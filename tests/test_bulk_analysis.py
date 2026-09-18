import argparse
import struct

import numpy as np
import pytest
from brass import AnalysisRunner
from brass.analyses import Bulk

PION_PDG = 211
PROTON_PDG = 2212

PION_MASS = 0.13957
PROTON_MASS = 0.93827

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
    pdg,
    mass,
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

    mt = np.sqrt(mass**2 + pt**2)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = mt * np.sinh(y)
    p0 = mt * np.cosh(y)

    particles = [
        [
            float(energy),
            float(px_i),
            float(py_i),
            float(pz_i),
            int(pdg),
        ]
        for energy, px_i, py_i, pz_i in zip(
            p0,
            px,
            py,
            pz,
        )
    ]

    return particles, y, pt, mt


def make_run(folder, seed, meta):
    folder.mkdir()

    (folder / "config.yaml").write_text(
        f"""
Meta:
  test: {meta}

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

    pions, pion_y, pion_pt, pion_mt = generate_particles(
        rng,
        N_PARTICLES,
        PION_PDG,
        PION_MASS,
        0.0,
        1.0,
        0.4,
    )

    protons, proton_y, proton_pt, proton_mt = generate_particles(
        rng,
        N_PARTICLES,
        PROTON_PDG,
        PROTON_MASS,
        0.5,
        0.4,
        0.6,
    )

    path = folder / "particles_binary.bin"

    with path.open("wb") as file:
        write_header(file)
        write_particle_block(
            file,
            pions + protons,
        )

    return {
        PION_PDG: {
            "y": pion_y,
            "pt": pion_pt,
            "mt": pion_mt,
        },
        PROTON_PDG: {
            "y": proton_y,
            "pt": proton_pt,
            "mt": proton_mt,
        },
    }


def histogram_mean(counts, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])

    return np.average(
        centers,
        weights=counts,
    )


def test_bulk_with_analysis_runner(tmp_path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run3 = tmp_path / "run3"

    # Same metadata -> these must merge.
    make_run(
        run1,
        seed=42,
        meta=1,
    )
    make_run(
        run2,
        seed=43,
        meta=1,
    )

    # Different metadata -> this must remain separate.
    make_run(
        run3,
        seed=44,
        meta=2,
    )

    args = argparse.Namespace(
        pdgs=[
            PION_PDG,
            PROTON_PDG,
        ],
    )

    runner = AnalysisRunner(
        Bulk,
        [
            run1,
            run2,
            run3,
        ],
        args=args,
        max_workers=2,
    )

    results = runner.run()

    # Meta.test=1 and Meta.test=2.
    assert len(results) == 2

    analyses = [entry["result"] for entry in results]

    assert all(isinstance(analysis, Bulk) for analysis in analyses)

    # run1 + run2
    merged = next(analysis for analysis in analyses if analysis.events == 2)

    # run3
    separate = next(analysis for analysis in analyses if analysis.events == 1)

    assert merged.pdgs == [
        PION_PDG,
        PROTON_PDG,
    ]

    assert separate.pdgs == [
        PION_PDG,
        PROTON_PDG,
    ]

    assert PION_PDG in merged.histograms
    assert PROTON_PDG in merged.histograms

    assert PION_PDG in separate.histograms
    assert PROTON_PDG in separate.histograms

    merged_pion_values = np.asarray(merged.histograms[PION_PDG].values)

    merged_proton_values = np.asarray(merged.histograms[PROTON_PDG].values)

    assert merged_pion_values.shape == (100, 100)
    assert merged_proton_values.shape == (100, 100)

    assert merged_pion_values.sum() > N_PARTICLES
    assert merged_pion_values.sum() <= 2 * N_PARTICLES

    assert merged_proton_values.sum() > N_PARTICLES
    assert merged_proton_values.sum() <= 2 * N_PARTICLES

    separate_pion_values = np.asarray(separate.histograms[PION_PDG].values)

    separate_proton_values = np.asarray(separate.histograms[PROTON_PDG].values)

    assert separate_pion_values.shape == (100, 100)
    assert separate_proton_values.shape == (100, 100)

    assert separate_pion_values.sum() > 0
    assert separate_pion_values.sum() <= N_PARTICLES

    assert separate_proton_values.sum() > 0
    assert separate_proton_values.sum() <= N_PARTICLES


def test_bulk_recovers_generated_distributions(tmp_path):
    run = tmp_path / "run"

    generated = make_run(
        run,
        seed=42,
        meta=1,
    )

    args = argparse.Namespace(
        pdgs=[
            PION_PDG,
            PROTON_PDG,
        ],
    )

    analysis = Bulk(
        run,
        args,
    )

    analysis.run()

    for pdg in [
        PION_PDG,
        PROTON_PDG,
    ]:
        hist = analysis.histograms[pdg]

        values = np.asarray(hist.values)

        assert values.shape == (100, 100)

        y_edges = np.asarray(hist.edges[0])
        mt_edges = np.asarray(hist.edges[1])

        # Project the 2D (y, mt) histogram onto each axis.
        y_counts = values.sum(axis=1)
        mt_counts = values.sum(axis=0)

        recovered_y_mean = histogram_mean(
            y_counts,
            y_edges,
        )

        recovered_mt_mean = histogram_mean(
            mt_counts,
            mt_edges,
        )

        # Only compare generated particles that lie inside the
        # histogram ranges. Out-of-range particles are not present
        # in the histogram.
        generated_y = generated[pdg]["y"]
        generated_mt = generated[pdg]["mt"]

        y_mask = (
            (generated_y >= y_edges[0])
            & (generated_y < y_edges[-1])
            & (generated_mt >= mt_edges[0])
            & (generated_mt < mt_edges[-1])
        )

        expected_y_mean = generated_y[y_mask].mean()

        expected_mt_mean = generated_mt[y_mask].mean()

        assert recovered_y_mean == pytest.approx(
            expected_y_mean,
            abs=0.02,
        )

        assert recovered_mt_mean == pytest.approx(
            expected_mt_mean,
            abs=0.02,
        )
