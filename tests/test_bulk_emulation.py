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
        observables=["hist2d"],
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

    direct = DataHandler(results, parameter_paths)
    loaded = DataHandler.from_file(path, parameter_paths)

    assert isinstance(loaded, DataHandler)
    np.testing.assert_allclose(loaded.parameters, direct.parameters)
    np.testing.assert_allclose(
        loaded.histograms[PION_PDG], direct.histograms[PION_PDG]
    )

    emulator = BulkEmulator.from_file(path, parameter_paths, n_components=2)
    assert isinstance(emulator, BulkEmulator)


# ---------------------------------------------------------------------------
# dN/dy emulation, persistence and the batched likelihood
# ---------------------------------------------------------------------------

PARAMETER_PATHS = {
    "y_mean": ("Meta", "y_mean"),
    "y_sigma": ("Meta", "y_sigma"),
    "pt_mean": ("Meta", "pt_mean"),
}

GRID = [
    (y_mean, y_sigma, pt_mean)
    for y_mean in (-0.5, 0.0, 0.5)
    for y_sigma in (0.8, 1.2)
    for pt_mean in (0.3, 0.5)
]


@pytest.fixture(scope="module")
def grid_results(tmp_path_factory):
    folder = tmp_path_factory.mktemp("grid")
    runs = []
    for i, (y_mean, y_sigma, pt_mean) in enumerate(GRID):
        run = folder / f"run{i}"
        make_run(run, seed=100 + i, y_mean=y_mean, y_sigma=y_sigma, pt_mean=pt_mean)
        runs.append(run)
    return AnalysisRunner(Bulk, runs, args=bulk_args([PION_PDG]), max_workers=2).run()


@pytest.fixture(scope="module")
def dndy_emulator(grid_results):
    return BulkEmulator(
        grid_results, PARAMETER_PATHS, observables=["dNdy", "hist2d"]
    ).fit()


def test_dndy_derived_from_hist2d(grid_results):
    emu = BulkEmulator(grid_results, PARAMETER_PATHS, observables=["hist2d"]).fit()
    target = [[0.1, 1.0, 0.4], [-0.2, 0.9, 0.35]]

    obs = emu.observables(PION_PDG)
    y, derived, std = emu.dNdy_with_std(target, PION_PDG)
    np.testing.assert_allclose(y, obs.y)
    np.testing.assert_allclose(derived, obs.dNdy(emu.predict(target, PION_PDG)))
    assert np.all(std > 0)


def test_dndy_reproduces_training_runs(dndy_emulator):
    truth = dndy_emulator.data.observable(PION_PDG, "dNdy")
    _, mean = dndy_emulator.dNdy(dndy_emulator.data.parameters, PION_PDG)
    assert np.max(np.abs(mean - truth)) < 0.1 * truth.max()


def test_dndy_only_emulator_derives_yield(grid_results):
    emu = BulkEmulator(grid_results, PARAMETER_PATHS).fit()
    assert emu.observable_names == ["dNdy"]

    target = [[0.0, 1.0, 0.4], [0.2, 0.9, 0.35]]
    _, dndy = emu.dNdy(target, PION_PDG)
    obs = emu.observables(PION_PDG)
    np.testing.assert_allclose(emu.yield_total(target, PION_PDG), dndy @ obs.dy)

    with pytest.raises(ValueError, match="cannot be obtained"):
        emu.dNdmt(target, PION_PDG)


def test_save_and_load(dndy_emulator, tmp_path):
    path = tmp_path / "emulator.npz"
    dndy_emulator.save(path)
    loaded = BulkEmulator.load(path)

    assert loaded.parameter_names == dndy_emulator.parameter_names
    assert loaded.data is None

    target = [[0.1, 1.0, 0.4]]
    for a, b in zip(
        dndy_emulator.dNdy_with_covariance(target, PION_PDG),
        loaded.dNdy_with_covariance(target, PION_PDG),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)

    with pytest.raises(RuntimeError):
        loaded.fit()


def test_validate(dndy_emulator):
    results = dndy_emulator.validate(holdout=0.25, observables=["dNdy"])
    result = results[(PION_PDG, "dNdy")]
    assert result["truth"].shape == result["mean"].shape == result["std"].shape
    assert len(result["parameters"]) == 3


def test_missing_pdg_is_reported():
    from types import SimpleNamespace

    from brass.emulation.bulk import DataHandler

    def run(pdgs):
        histo = SimpleNamespace(values=np.ones((2, 2)), edges=[[0, 1, 2], [0, 1, 2]])
        analysis = SimpleNamespace(
            config={"Meta": {"a": 1.0}},
            events=1,
            folder_path="run",
            histograms={pdg: histo for pdg in pdgs},
        )
        return {"result": analysis}

    with pytest.raises(ValueError, match="missing"):
        DataHandler([run([211, 321]), run([211])], {"a": ("Meta", "a")})
    with pytest.raises(ValueError, match="missing"):
        DataHandler([run([211]), run([211, 321])], {"a": ("Meta", "a")})


def _manual_log_likelihood(emu, theta, data, covariance):
    y, mean, cov = emu.dNdy_with_covariance(theta, PION_PDG)
    W = np.array(
        [np.interp(data["x"], y, np.eye(len(y))[j]) for j in range(len(y))]
    ).T
    m = W @ mean[0]
    C = np.diag(data["yerr"] ** 2)
    if covariance == "full":
        C = C + W @ cov[0] @ W.T
    r = m - data["y"]
    chi2 = r @ np.linalg.solve(C, r)
    return -0.5 * (chi2 + np.linalg.slogdet(C)[1] + len(r) * np.log(2 * np.pi))


def test_emulator_likelihood_matches_manual(dndy_emulator):
    from brass.emulation import EmulatorLikelihood

    rng = np.random.default_rng(0)
    x = np.linspace(-2.5, 2.5, 9)
    _, truth = dndy_emulator.dNdy([[0.0, 1.0, 0.4]], PION_PDG)
    obs = dndy_emulator.observables(PION_PDG)
    y = np.interp(x, obs.y, truth[0]) * (1 + 0.02 * rng.normal(size=len(x)))
    data = {"x": np.append(x, 10.0), "y": np.append(y, 1.0), "yerr": np.append(0.03 * y, 0.1)}

    thetas = np.array([[0.0, 1.0, 0.4], [0.3, 0.9, 0.45], [-0.4, 1.1, 0.3]])

    for covariance in ("full", "none"):
        like = EmulatorLikelihood(dndy_emulator, covariance=covariance)
        like.add(PION_PDG, data["x"], data["y"], data["yerr"])
        assert like.n_points == len(x)  # point at x=10 is outside the emulator

        inside = {k: v[:-1] for k, v in data.items()}
        expected = [
            _manual_log_likelihood(dndy_emulator, [t], inside, covariance)
            for t in thetas
        ]
        np.testing.assert_allclose(like(thetas), expected, rtol=1e-8)

    # The true parameters fit best.
    assert np.argmax(like(thetas)) == 0


def test_emulator_likelihood_fixed_parameters(dndy_emulator):
    from brass.emulation import EmulatorLikelihood

    x = np.linspace(-2, 2, 5)
    like = EmulatorLikelihood(dndy_emulator).add(
        PION_PDG, x, np.ones(5), np.full(5, 0.1), fixed={"y_sigma": 1.0}
    )
    assert like.parameters == ["y_mean", "pt_mean"]

    full = EmulatorLikelihood(dndy_emulator, covariance="full").add(
        PION_PDG, x, np.ones(5), np.full(5, 0.1)
    )
    np.testing.assert_allclose(like([[0.1, 0.4]]), full([[0.1, 1.0, 0.4]]))

    with pytest.raises(ValueError, match="columns"):
        like([[0.1, 1.0, 0.4]])
    with pytest.raises(KeyError):
        EmulatorLikelihood(dndy_emulator).add(
            PION_PDG, x, np.ones(5), np.full(5, 0.1), fixed={"energy": 1.0}
        )
