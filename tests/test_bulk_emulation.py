from types import SimpleNamespace

import numpy as np
import pytest

# Emulation needs the optional extra: pip install -e ".[emulation]"
pytest.importorskip("sklearn")

from brass import AnalysisRunner
from brass.analyses import Bulk
from brass.emulation import EmulatorLikelihood
from brass.emulation.bulk import (
    BulkEmulator,
    DataHandler,
    MeanPtXf,
    MtSpectrum,
    XfSpectrum,
)
from brass.emulation.bulk.observables import Observables
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


# ---------------------------------------------------------------------------
# mT reduction of the histograms (DataHandler.reduce)
# ---------------------------------------------------------------------------

Y_EDGES = np.linspace(-2.0, 2.0, 9)
MT_EDGES = np.linspace(0.0, 2.0, 11)


def random_hist(seed=0, shape=(8, 10)):
    return np.random.default_rng(seed).uniform(0.5, 2.0, size=shape)


def handler(n_runs=3, n_events=4):
    """DataHandler from fake Bulk results with known counts."""
    rng = np.random.default_rng(1)
    runs, counts = [], []
    for i in range(n_runs):
        c = rng.integers(0, 50, size=(len(Y_EDGES) - 1, len(MT_EDGES) - 1))
        counts.append(c)
        histo = SimpleNamespace(values=c, edges=[Y_EDGES, MT_EDGES])
        analysis = SimpleNamespace(
            config={"Meta": {"a": float(i)}},
            events=n_events,
            folder_path=f"run{i}",
            histograms={PION_PDG: histo},
        )
        runs.append({"result": analysis})
    return DataHandler(runs, {"a": ("Meta", "a")}), np.array(counts), n_events


def test_reduce_crops_and_merges_mt():
    data, counts, events = handler()
    reduced = data.reduce(mt_max=1.2, mt_group=2)

    np.testing.assert_allclose(reduced.mt_edges[PION_PDG], [0.0, 0.4, 0.8, 1.2])
    np.testing.assert_allclose(reduced.y_edges[PION_PDG], Y_EDGES)
    H = reduced.histograms[PION_PDG]
    assert H.shape == (3, 8, 3)

    # Particle numbers per (y, merged mT) bin are those of the raw counts.
    dy = np.diff(Y_EDGES)[:, None]
    expected = counts[..., :6].reshape(3, 8, 3, 2).sum(-1) / events
    np.testing.assert_allclose(H * dy * 0.4, expected)

    # The original handler is unchanged.
    assert data.histograms[PION_PDG].shape == (3, 8, 10)
    assert len(data.mt_edges[PION_PDG]) == 11


def test_reduce_keeps_dndy_below_cut():
    data, _, _ = handler()
    reduced = data.reduce(mt_max=1.0, mt_group=5)
    cropped = data.reduce(mt_max=1.0)
    np.testing.assert_allclose(
        reduced.observable(PION_PDG, "dNdy"), cropped.observable(PION_PDG, "dNdy")
    )
    full = data.reduce(mt_group=2)
    np.testing.assert_allclose(
        full.observable(PION_PDG, "dNdy"), data.observable(PION_PDG, "dNdy")
    )


def test_reduce_rejects_bad_settings():
    data, _, _ = handler()
    with pytest.raises(ValueError, match="groups of 3"):
        data.reduce(mt_max=1.0, mt_group=3)
    with pytest.raises(ValueError, match="no mT bin"):
        data.reduce(mt_max=0.1)
    with pytest.raises(ValueError, match="positive"):
        data.reduce(mt_group=0)


# ---------------------------------------------------------------------------
# mT spectrum derived from the 2D histogram
# ---------------------------------------------------------------------------


def test_mt_spectrum_averages_rapidity_window():
    obs = Observables(Y_EDGES, MT_EDGES)
    H = random_hist()

    # Window [-0.75, 0.5]: half of bin [-1, -0.5], all of [-0.5, 0] and [0, 0.5].
    mt, A = MtSpectrum(-0.75, 0.5).build(obs)
    expected = (0.5 * 0.5 * H[2] + 0.5 * H[3] + 0.5 * H[4]) / 1.25
    np.testing.assert_allclose(mt, obs.mt)
    np.testing.assert_allclose(A @ H.ravel(), expected)

    with pytest.raises(ValueError, match="outside"):
        MtSpectrum(-3.0, 0.0).build(obs)
    with pytest.raises(ValueError, match="empty"):
        MtSpectrum(0.5, 0.5).build(obs)


# ---------------------------------------------------------------------------
# Reduced hist2d emulator: spectra, likelihood and validation
# ---------------------------------------------------------------------------

MT_MAX = 2.0
SPECS = [MtSpectrum(-0.5, 0.5), MtSpectrum(0.0, 0.25)]


@pytest.fixture(scope="module")
def reduced_emulator(grid_results):
    return BulkEmulator(
        grid_results,
        PARAMETER_PATHS,
        observables=["hist2d"],
        mt_max=MT_MAX,
        mt_group=2,
    ).fit()


def test_emulator_uses_reduced_histograms(grid_results, reduced_emulator):
    full = DataHandler(grid_results, PARAMETER_PATHS)
    edges = full.mt_edges[PION_PDG]
    n_kept = int(np.sum(edges[1:] <= MT_MAX * (1 + 1e-12)))

    np.testing.assert_allclose(
        reduced_emulator.mt_edges[PION_PDG], edges[: n_kept + 1 : 2]
    )
    n_y = len(reduced_emulator.y_edges[PION_PDG]) - 1
    assert reduced_emulator.data.histograms[PION_PDG].shape[1:] == (n_y, n_kept // 2)
    assert reduced_emulator.predict([[0.0, 1.0, 0.4]], PION_PDG).shape == (
        1,
        n_y,
        n_kept // 2,
    )


@pytest.mark.parametrize("spec", SPECS, ids=str)
def test_spectra_derived_from_hist2d(reduced_emulator, spec):
    theta = [[0.1, 1.0, 0.4], [-0.2, 0.9, 0.35]]
    obs = reduced_emulator.observables(PION_PDG)
    axis, A = spec.build(obs)

    mean, cov = reduced_emulator.predict_observable(
        theta, PION_PDG, spec, return_cov=True
    )
    hist = reduced_emulator.predict(theta, PION_PDG).reshape(2, -1)
    np.testing.assert_allclose(reduced_emulator.axis(PION_PDG, spec), axis)
    np.testing.assert_allclose(mean, hist @ A.T, rtol=1e-10, atol=1e-14)
    assert cov.shape == (2, len(axis), len(axis))

    # Training values of the spectrum come from the same matrix.
    np.testing.assert_allclose(
        reduced_emulator.data.observable(PION_PDG, spec),
        reduced_emulator.data.observable(PION_PDG, "hist2d") @ A.T,
    )


def test_spectra_need_hist2d(grid_results):
    emu = BulkEmulator(grid_results, PARAMETER_PATHS, observables=["dNdy"]).fit()
    with pytest.raises(ValueError, match="cannot be obtained"):
        emu.predict_observable([[0.0, 1.0, 0.4]], PION_PDG, SPECS[0])


@pytest.mark.parametrize("spec", SPECS, ids=str)
def test_likelihood_with_spectrum_matches_manual(reduced_emulator, spec):
    axis = reduced_emulator.axis(PION_PDG, spec)
    x = np.linspace(axis[1], axis[-2], 6)
    truth = reduced_emulator.predict_observable([[0.0, 1.0, 0.4]], PION_PDG, spec)[0]
    y = np.interp(x, axis, truth)
    yerr = 0.05 * np.abs(y) + 1e-3

    like = EmulatorLikelihood(reduced_emulator).add(
        PION_PDG, x, y, yerr, observable=spec
    )
    assert str(spec) in like.datasets[0]["label"]

    thetas = np.array([[0.0, 1.0, 0.4], [0.3, 0.9, 0.45], [-0.3, 1.1, 0.3]])
    W = np.array([np.interp(x, axis, np.eye(len(axis))[j]) for j in range(len(axis))]).T
    expected = []
    for theta in thetas:
        mean, cov = reduced_emulator.predict_observable(
            [theta], PION_PDG, spec, return_cov=True
        )
        C = np.diag(yerr**2) + W @ cov[0] @ W.T
        r = W @ mean[0] - y
        expected.append(
            -0.5
            * (
                r @ np.linalg.solve(C, r)
                + np.linalg.slogdet(C)[1]
                + len(r) * np.log(2 * np.pi)
            )
        )
    np.testing.assert_allclose(like(thetas), expected, rtol=1e-8)
    assert np.argmax(like(thetas)) == 0


def test_reduced_emulator_save_and_load(reduced_emulator, tmp_path):
    path = tmp_path / "emulator.npz"
    reduced_emulator.save(path)
    loaded = BulkEmulator.load(path)
    np.testing.assert_allclose(
        loaded.mt_edges[PION_PDG], reduced_emulator.mt_edges[PION_PDG]
    )

    spec = SPECS[0]
    theta = [[0.1, 1.0, 0.4]]
    np.testing.assert_allclose(
        loaded.predict_observable(theta, PION_PDG, spec),
        reduced_emulator.predict_observable(theta, PION_PDG, spec),
        rtol=1e-10,
    )


def test_validate_derives_spectra_from_hist2d(reduced_emulator):
    """Spectra are validated with the hist2d emulator they come from."""
    results = reduced_emulator.validate(
        holdout=0.25, observables=["hist2d", "dNdy", *SPECS]
    )
    hist = results[(PION_PDG, "hist2d")]
    obs = reduced_emulator.observables(PION_PDG)

    for name in ["dNdy", *SPECS]:
        r = results[(PION_PDG, name)]
        _, A = reduced_emulator.linear_model(PION_PDG, name)
        np.testing.assert_allclose(r["parameters"], hist["parameters"])
        np.testing.assert_allclose(r["truth"], hist["truth"] @ A.T, atol=1e-14)
        np.testing.assert_allclose(
            r["mean"], hist["mean"] @ A.T, rtol=1e-10, atol=1e-14
        )
        assert r["mean"].shape == r["std"].shape == (3, A.shape[0])
        assert np.all(r["std"] > 0)

    # The held-out spectra are predicted well where they are large.
    for spec in SPECS:
        r = results[(PION_PDG, spec)]
        big = r["truth"] > 0.2 * r["truth"].max()
        rel = np.abs(r["mean"] - r["truth"])[big] / r["truth"][big]
        assert np.median(rel) < 0.1, (str(spec), np.median(rel))
        z = r["residuals"][big]
        # Standardised residuals have roughly unit width: σ is the right size.
        assert np.all(np.isfinite(z)) and 0.3 < np.std(z) < 2, (str(spec), np.std(z))

    assert obs.y.shape == results[(PION_PDG, "dNdy")]["mean"].shape[1:]


def test_validate_rejects_underivable_spectrum(grid_results):
    emu = BulkEmulator(grid_results, PARAMETER_PATHS, observables=["dNdy"])
    with pytest.raises(ValueError, match="cannot be obtained"):
        emu.validate(observables=[SPECS[0]])


# ---------------------------------------------------------------------------
# x_F observables from the (x_F, pT) histogram (Bulk --xf-bins)
# ---------------------------------------------------------------------------

SQRTS = 6.0
XF_ARGS = ("--xf-bins", "40", "--pt-bins", "30", "--pt-range", "0", "3")
XF_EDGES = np.linspace(-0.2, 0.2, 9)


def xf_hist(seed=0, n_xf=40, n_pt=30):
    return np.random.default_rng(seed).uniform(0.5, 2.0, size=(3, n_xf, n_pt))


def test_xf_observables_from_histogram():
    xf_edges = np.linspace(-1, 1, 41)
    pt_edges = np.linspace(0, 3, 31)
    H = xf_hist()
    counts = H * np.diff(pt_edges) * 0.05  # per event, per (x_F, pT) bin
    pt = 0.5 * (pt_edges[1:] + pt_edges[:-1])

    # Two stored bins per requested bin.
    edges = np.linspace(-0.5, 0.5, 11)
    dndxf = XfSpectrum(edges).values(H, xf_edges, pt_edges)
    np.testing.assert_allclose(
        dndxf, counts[:, 10:30].reshape(3, 10, 2, 30).sum((2, 3)) / 0.1
    )
    np.testing.assert_allclose(XfSpectrum(edges).axis(), np.linspace(-0.45, 0.45, 10))

    mean_pt = MeanPtXf(edges).values(H, xf_edges, pt_edges)
    grouped = counts[:, 10:30].reshape(3, 10, 2, 30).sum(2)
    np.testing.assert_allclose(mean_pt, grouped @ pt / grouped.sum(-1))

    # Uneven bins are fine as long as the edges are on the stored grid.
    uneven = XfSpectrum([-1.0, -0.2, 0.0, 0.85]).values(H, xf_edges, pt_edges)
    np.testing.assert_allclose(
        uneven[:, 0] * 0.8, counts[:, :16].sum((1, 2)), rtol=1e-12
    )


def test_xf_observables_reject_bad_edges():
    xf_edges = np.linspace(-1, 1, 41)
    pt_edges = np.linspace(0, 3, 31)
    with pytest.raises(ValueError, match="increasing"):
        XfSpectrum([0.1, 0.0])
    with pytest.raises(ValueError, match="increasing"):
        MeanPtXf([0.1])
    with pytest.raises(ValueError, match="not an edge"):
        XfSpectrum([0.0, 0.03]).values(xf_hist(), xf_edges, pt_edges)

    H = xf_hist()
    H[1, 20] = 0.0
    with pytest.raises(ValueError, match="no particles"):
        MeanPtXf([0.0, 0.05, 0.1]).values(H, xf_edges, pt_edges)


@pytest.fixture(scope="module")
def xf_results(tmp_path_factory):
    folder = tmp_path_factory.mktemp("xf")
    runs = []
    for i, (y_mean, y_sigma, pt_mean) in enumerate(GRID):
        run = folder / f"run{i}"
        pions = generate_particles(
            np.random.default_rng(300 + i), N_PARTICLES, pdg=PION_PDG,
            mass=PION_MASS, y_mean=y_mean, y_sigma=y_sigma, pt_mean=pt_mean,
        )
        config = {
            "Meta": {"y_mean": y_mean, "y_sigma": y_sigma, "pt_mean": pt_mean},
            "Modi": {"Collider": {"Sqrtsnn": SQRTS,
                                  "Calculation_Frame": "center of mass"}},
        }
        with write_run(run, QUANTITIES, config=config) as w:
            w.particles(pions)
        runs.append((run, pions))
    args = analysis_args(Bulk, "--pdgs", str(PION_PDG), *XF_ARGS)
    results = AnalysisRunner(Bulk, [r for r, _ in runs], args=args, max_workers=2).run()
    return results, [p for _, p in runs]


def test_data_handler_reads_xf_histograms(xf_results):
    results, particles = xf_results
    data = DataHandler(results, PARAMETER_PATHS)
    np.testing.assert_allclose(data.xf_edges[PION_PDG], np.linspace(-1, 1, 41))
    np.testing.assert_allclose(data.pt_edges[PION_PDG], np.linspace(0, 3, 31))
    assert data.xf_histograms[PION_PDG].shape == (len(GRID), 40, 30)

    # Exact x_F binning: compare with the generated particles (one event per run).
    order = [GRID.index(tuple(p)) for p in data.parameters]
    for run, k in enumerate(order):
        pions = particles[k]
        xf = 2 * pions["pz"] / SQRTS
        inside = (np.abs(xf) < 0.2) & (pions["pt"] < 3)
        dndxf = data.observable(PION_PDG, XfSpectrum(XF_EDGES))[run]
        expected, _ = np.histogram(xf[inside], XF_EDGES)
        np.testing.assert_allclose(dndxf * 0.05, expected)

        mean_pt = data.observable(PION_PDG, MeanPtXf(XF_EDGES))[run]
        k_bin = np.digitize(xf[inside], XF_EDGES) - 1
        exact = np.bincount(k_bin, pions["pt"][inside]) / np.bincount(k_bin)
        # pT bin centres of 0.1 GeV bins: small bias only.
        np.testing.assert_allclose(mean_pt, exact, atol=0.01)


def test_data_handler_without_xf_histograms(grid_results):
    data = DataHandler(grid_results, PARAMETER_PATHS)
    assert data.xf_histograms == {}
    with pytest.raises(ValueError, match="--xf-bins"):
        data.observable(PION_PDG, MeanPtXf(XF_EDGES))


@pytest.fixture(scope="module")
def xf_emulator(xf_results):
    return BulkEmulator(
        xf_results[0],
        PARAMETER_PATHS,
        observables=["dNdy", XfSpectrum(XF_EDGES), MeanPtXf(XF_EDGES)],
    ).fit()


def test_xf_emulator_fits_observables_directly(xf_emulator):
    for spec in (XfSpectrum(XF_EDGES), MeanPtXf(XF_EDGES)):
        _, A = xf_emulator.linear_model(PION_PDG, spec)
        assert A is None
        np.testing.assert_allclose(xf_emulator.axis(PION_PDG, spec), spec.axis())

        truth = xf_emulator.data.observable(PION_PDG, spec)
        mean = xf_emulator.predict_observable(xf_emulator.data.parameters, PION_PDG, spec)
        assert np.median(np.abs(mean - truth) / truth) < 0.05, str(spec)

    # <pT> follows the generated mean pT.
    spec = MeanPtXf(XF_EDGES)
    low, high = xf_emulator.predict_observable(
        [[0.0, 1.0, 0.3], [0.0, 1.0, 0.5]], PION_PDG, spec
    )
    assert np.all(high > 1.2 * low)


def test_xf_emulator_likelihood_and_validate(xf_emulator):
    spec = MeanPtXf(XF_EDGES)
    x = spec.axis()
    truth = xf_emulator.predict_observable([[0.0, 1.0, 0.4]], PION_PDG, spec)[0]
    like = EmulatorLikelihood(xf_emulator).add(
        PION_PDG, x, truth, np.full(len(x), 0.01), observable=spec
    )
    thetas = np.array([[0.0, 1.0, 0.4], [0.0, 1.0, 0.3], [0.0, 1.0, 0.5]])
    assert np.argmax(like(thetas)) == 0
    assert str(spec) in like.datasets[0]["label"]

    results = xf_emulator.validate(holdout=0.25, observables=[spec])
    r = results[(PION_PDG, spec)]
    assert r["mean"].shape == r["truth"].shape == (3, len(x))
    assert np.all(np.isfinite(r["residuals"]))


def test_xf_emulator_save_and_load(xf_emulator, tmp_path):
    path = tmp_path / "emulator.npz"
    xf_emulator.save(path)
    loaded = BulkEmulator.load(path)
    assert loaded.observable_names == xf_emulator.observable_names

    theta = [[0.1, 1.0, 0.4]]
    for spec in loaded.observable_names:
        np.testing.assert_allclose(
            loaded.predict_observable(theta, PION_PDG, spec, return_cov=True)[1],
            xf_emulator.predict_observable(theta, PION_PDG, spec, return_cov=True)[1],
            rtol=1e-10,
        )


def test_load_format_1(dndy_emulator, tmp_path):
    """Emulators saved before the x_F observables still load."""
    import json

    path = tmp_path / "new.npz"
    dndy_emulator.save(path)
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    meta = json.loads(str(arrays["meta"]))
    old = {"meta": None}
    for key, value in arrays.items():
        if key.startswith("model/"):
            _, i, rest = key.split("/", 2)
            pdg, name = meta["models"][int(i)]
            key = f"model/{pdg}/{name}/{rest}"
        old[key] = value
    meta["format"] = 1
    old["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(tmp_path / "old.npz", **old)

    loaded = BulkEmulator.load(tmp_path / "old.npz")
    theta = [[0.1, 1.0, 0.4]]
    np.testing.assert_allclose(
        loaded.dNdy(theta, PION_PDG)[1], dndy_emulator.dNdy(theta, PION_PDG)[1]
    )
