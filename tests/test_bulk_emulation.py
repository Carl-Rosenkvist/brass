import numpy as np
import pytest

# Emulation needs the optional extra: pip install -e ".[emulation]"
pytest.importorskip("sklearn")

from brass import AnalysisRunner
from brass.analyses import Bulk
from brass.emulation.bulk import BulkEmulator
from brass.testing import analysis_args, generate_particles, write_run

PION_PDG = 211
PION_MASS = 0.13957

N_PARTICLES = 10_000

QUANTITIES = ["p0", "px", "py", "pz", "pdg"]


def bulk_args(pdgs):
    return analysis_args(Bulk, "--pdgs", *map(str, pdgs))


def make_run(
    folder,
    seed,
    y_mean,
    y_sigma,
    pt_mean,
):
    rng = np.random.default_rng(seed)

    pions = generate_particles(
        rng,
        N_PARTICLES,
        pdg=PION_PDG,
        mass=PION_MASS,
        y_mean=y_mean,
        y_sigma=y_sigma,
        pt_mean=pt_mean,
    )

    meta = {"y_mean": y_mean, "y_sigma": y_sigma, "pt_mean": pt_mean}

    with write_run(folder, QUANTITIES, config={"Meta": meta}) as w:
        w.particles(pions)


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
        args=bulk_args([PION_PDG]),
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


def test_from_file(tmp_path):
    from brass.emulation.bulk.data import DataHandler

    runs = []

    for i, (y_mean, pt_mean) in enumerate(
        [(-0.5, 0.3), (-0.5, 0.5), (0.5, 0.3), (0.5, 0.5)]
    ):
        run = tmp_path / f"run{i}"
        make_run(run, seed=i, y_mean=y_mean, y_sigma=1.0, pt_mean=pt_mean)
        runs.append(run)

    runner = AnalysisRunner(Bulk, runs, args=bulk_args([PION_PDG]), max_workers=1)
    path = tmp_path / "bulk.pkl"
    results = runner.run_and_save(path)

    parameter_paths = {
        "y_mean": ("Meta", "y_mean"),
        "pt_mean": ("Meta", "pt_mean"),
    }

    direct = DataHandler(results, parameter_paths, n_components=2)
    loaded = DataHandler.from_file(path, parameter_paths, n_components=2)

    assert isinstance(loaded, DataHandler)
    np.testing.assert_allclose(loaded.parameters, direct.parameters)
    np.testing.assert_allclose(
        loaded.histograms[PION_PDG], direct.histograms[PION_PDG]
    )

    emulator = BulkEmulator.from_file(path, parameter_paths, n_components=2)
    assert isinstance(emulator, BulkEmulator)
