import numpy as np
import pytest
from brass import AnalysisRunner
from brass.analyses import Bulk
from brass.testing import (
    analysis_args,
    concatenate_particles,
    generate_particles,
    write_run,
)

PION_PDG = 211
PROTON_PDG = 2212

PION_MASS = 0.13957
PROTON_MASS = 0.93827

N_PARTICLES = 10_000

QUANTITIES = ["p0", "px", "py", "pz", "pdg"]


def bulk_args(pdgs):
    return analysis_args(Bulk, "--pdgs", *map(str, pdgs))


def make_run(folder, seed, meta):
    rng = np.random.default_rng(seed)

    pions = generate_particles(
        rng,
        N_PARTICLES,
        pdg=PION_PDG,
        mass=PION_MASS,
        y_mean=0.0,
        y_sigma=1.0,
        pt_mean=0.4,
    )

    protons = generate_particles(
        rng,
        N_PARTICLES,
        pdg=PROTON_PDG,
        mass=PROTON_MASS,
        y_mean=0.5,
        y_sigma=0.4,
        pt_mean=0.6,
    )

    with write_run(folder, QUANTITIES, config={"Meta": {"test": meta}}) as w:
        w.particles(concatenate_particles(pions, protons))

    return {PION_PDG: pions, PROTON_PDG: protons}


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

    args = bulk_args([PION_PDG, PROTON_PDG])

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

    assert merged_pion_values.sum() > N_PARTICLES
    assert merged_pion_values.sum() <= 2 * N_PARTICLES

    assert merged_proton_values.sum() > N_PARTICLES
    assert merged_proton_values.sum() <= 2 * N_PARTICLES

    separate_pion_values = np.asarray(separate.histograms[PION_PDG].values)

    separate_proton_values = np.asarray(separate.histograms[PROTON_PDG].values)

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

    args = bulk_args([PION_PDG, PROTON_PDG])

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
