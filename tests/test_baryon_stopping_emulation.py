from types import SimpleNamespace

import numpy as np
import pytest

# Emulation needs the optional extra: pip install -e ".[emulation]"
pytest.importorskip("sklearn")

from brass import AnalysisRunner
from brass.analyses import BaryonStopping, Bulk
from brass.emulation import EmulatorLikelihood
from brass.emulation.baryon_stopping import BaryonStoppingData, BaryonStoppingEmulator
from brass.emulation.bulk import BulkEmulator
from brass.testing import analysis_args, generate_particles, write_run

PION_PDG = 211
PION_MASS = 0.13957
SQRTS = 6.0
N_PARTICLES = 10_000

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
def runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("bs")
    out = []
    for i, (y_mean, y_sigma, pt_mean) in enumerate(GRID):
        run = folder / f"run{i}"
        pions = generate_particles(
            np.random.default_rng(400 + i), N_PARTICLES, pdg=PION_PDG,
            mass=PION_MASS, y_mean=y_mean, y_sigma=y_sigma, pt_mean=pt_mean,
        )
        config = {
            "Meta": {"y_mean": y_mean, "y_sigma": y_sigma, "pt_mean": pt_mean},
            "Modi": {"Collider": {"Sqrtsnn": SQRTS, "Calculation_Frame": "center of mass"}},
        }
        with write_run(run, ["p0", "px", "py", "pz", "pdg"], config=config) as w:
            w.particles(pions)
        out.append((run, pions))
    return out


@pytest.fixture(scope="module")
def results(runs):
    args = analysis_args(BaryonStopping, "--pdgs", str(PION_PDG), "--xf-bins", "40")
    return AnalysisRunner(BaryonStopping, [r for r, _ in runs], args=args).run()


@pytest.fixture(scope="module")
def emulator(results):
    return BaryonStoppingEmulator(
        results, PARAMETER_PATHS, observables=["dNdxF", "meanpT_xF", "dNdy", "meanmT_y"]
    ).fit()


# -- training data --------------------------------------------------------------


def matching(centres, axis):
    return np.isclose(centres[:, None], axis[None, :], rtol=0, atol=1e-9).any(axis=1)


def test_data_matches_particles(results, runs):
    data = BaryonStoppingData(results, PARAMETER_PATHS)
    pions = {tuple(p): particles for p, (_, particles) in zip(GRID, runs)}

    xf_edges = np.linspace(-1, 1, 41)
    y_edges = np.linspace(-4, 4, 41)
    np.testing.assert_allclose(data.xf_edges[PION_PDG], xf_edges)

    mean_pt = data.observable(PION_PDG, "meanpT_xF")
    mean_mt = data.observable(PION_PDG, "meanmT_y")
    xf_axis = data.axis(PION_PDG, "meanpT_xF")
    y_axis = data.axis(PION_PDG, "meanmT_y")
    assert mean_pt.shape == (len(GRID), len(xf_axis))

    for run, parameters in enumerate(data.parameters):
        p = pions[tuple(parameters)]
        xf = 2 * p["pz"] / SQRTS
        counts, _ = np.histogram(xf, xf_edges)
        np.testing.assert_allclose(
            data.observable(PION_PDG, "dNdxF")[run], counts / 0.05
        )
        np.testing.assert_allclose(
            data.observable(PION_PDG, "dNdy")[run], np.histogram(p["y"], y_edges)[0] / 0.2
        )

        inside = np.abs(xf) < 1  # generated particles are not limited by sqrt(s)
        k = np.digitize(xf[inside], xf_edges) - 1
        exact = np.bincount(k, p["pt"][inside], minlength=40) / np.maximum(counts, 1)
        centres = 0.5 * (xf_edges[1:] + xf_edges[:-1])
        np.testing.assert_allclose(mean_pt[run], exact[matching(centres, xf_axis)])

        inside = np.abs(p["y"]) < 4
        k = np.digitize(p["y"][inside], y_edges) - 1
        n = np.bincount(k, minlength=40)
        exact = np.bincount(k, p["mt"][inside], minlength=40) / np.maximum(n, 1)
        centres = 0.5 * (y_edges[1:] + y_edges[:-1])
        np.testing.assert_allclose(mean_mt[run], exact[matching(centres, y_axis)])


def fake_results(counts):
    """BaryonStopping-like results with given x_F counts per run (2 x_F bins)."""

    def histogram(values, edges):
        return SimpleNamespace(values=np.asarray(values, float), edges=[edges])

    out = []
    for i, n in enumerate(counts):
        n = np.asarray(n, float)
        analysis = SimpleNamespace(
            config={"Meta": {"a": float(i)}},
            events=2,
            folder_path=f"run{i}",
            sqrts=10.0,
            histograms={
                "xf": {PION_PDG: histogram(n, [-5.0, 0.0, 5.0])},
                "xf_pt": {PION_PDG: histogram(0.4 * n, [-5.0, 0.0, 5.0])},
                "y": {PION_PDG: histogram([1.0], [-1.0, 1.0])},
                "y_mt": {PION_PDG: histogram([0.5], [-1.0, 1.0])},
            },
        )
        out.append({"result": analysis})
    return out


def test_mean_uses_bins_filled_in_every_run():
    data = BaryonStoppingData(fake_results([[3, 4], [0, 5]]), {"a": ("Meta", "a")})
    np.testing.assert_allclose(data.xf_edges[PION_PDG], [-1, 0, 1])
    np.testing.assert_allclose(data.observable(PION_PDG, "dNdxF"), [[1.5, 2.0], [0, 2.5]])
    np.testing.assert_allclose(data.axis(PION_PDG, "meanpT_xF"), [0.5])
    np.testing.assert_allclose(data.observable(PION_PDG, "meanpT_xF"), [[0.4], [0.4]])


def test_data_rejects_bulk_results(runs):
    args = analysis_args(Bulk, "--pdgs", str(PION_PDG))
    bulk = AnalysisRunner(Bulk, [r for r, _ in runs[:2]], args=args).run()
    with pytest.raises(TypeError, match="not a BaryonStopping result"):
        BaryonStoppingData(bulk, PARAMETER_PATHS)


# -- emulator -------------------------------------------------------------------


def test_emulator_reproduces_training_runs(emulator):
    for name in emulator.observable_names:
        truth = emulator.data.observable(PION_PDG, name)
        mean = emulator.predict_observable(emulator.data.parameters, PION_PDG, name)
        assert mean.shape == truth.shape
        np.testing.assert_allclose(emulator.axis(PION_PDG, name), emulator.data.axis(PION_PDG, name))
        big = np.abs(truth) > 0.2 * np.abs(truth).max()
        assert np.median(np.abs(mean - truth)[big] / np.abs(truth)[big]) < 0.05, name


def test_mean_pt_follows_parameter(emulator):
    low, high = emulator.predict_observable(
        [[0.0, 1.0, 0.3], [0.0, 1.0, 0.5]], PION_PDG, "meanpT_xF"
    )
    assert np.all(high > 1.2 * low)


def test_unknown_or_unfitted_observables(results, emulator):
    with pytest.raises(ValueError, match="unknown observables"):
        BaryonStoppingEmulator(results, PARAMETER_PATHS, observables=["hist2d"])
    only_xf = BaryonStoppingEmulator(results, PARAMETER_PATHS, observables=["dNdxF"]).fit()
    with pytest.raises(ValueError, match="not fitted"):
        only_xf.axis(PION_PDG, "meanpT_xF")
    with pytest.raises(ValueError, match="cannot be obtained"):
        only_xf.predict_observable([[0.0, 1.0, 0.4]], PION_PDG, "meanpT_xF")
    with pytest.raises(KeyError):
        emulator.axis(2212, "dNdxF")


def test_validate(emulator):
    r = emulator.validate(holdout=0.25, observables=["meanpT_xF"])[(PION_PDG, "meanpT_xF")]
    assert r["mean"].shape == r["truth"].shape == (3, len(emulator.axis(PION_PDG, "meanpT_xF")))
    assert np.all(np.isfinite(r["residuals"]))


def test_save_and_load(emulator, tmp_path):
    path = tmp_path / "bs.npz"
    emulator.save(path)
    loaded = BaryonStoppingEmulator.load(path)

    assert loaded.data is None
    assert loaded.pdgs == emulator.pdgs
    theta = [[0.1, 1.0, 0.4]]
    for name in emulator.observable_names:
        np.testing.assert_allclose(loaded.axis(PION_PDG, name), emulator.axis(PION_PDG, name))
        for a, b in zip(
            loaded.predict_observable(theta, PION_PDG, name, return_cov=True),
            emulator.predict_observable(theta, PION_PDG, name, return_cov=True),
        ):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)

    with pytest.raises(TypeError, match="BaryonStoppingEmulator"):
        BulkEmulator.load(path)


# -- likelihood with two emulators ------------------------------------------------


def test_likelihood_combines_emulators(runs, emulator):
    args = analysis_args(Bulk, "--pdgs", str(PION_PDG))
    bulk_results = AnalysisRunner(Bulk, [r for r, _ in runs], args=args).run()
    bulk = BulkEmulator(bulk_results, PARAMETER_PATHS).fit()

    theta_true = [[0.0, 1.0, 0.4]]
    y = np.linspace(-1.5, 1.5, 7)
    _, dndy = bulk.dNdy(theta_true, PION_PDG)
    dndy = np.interp(y, bulk.axis(PION_PDG, "dNdy"), dndy[0])
    xf = emulator.axis(PION_PDG, "meanpT_xF")
    mean_pt = emulator.predict_observable(theta_true, PION_PDG, "meanpT_xF")[0]

    both = EmulatorLikelihood(bulk)
    both.add(PION_PDG, y, dndy, 0.05 * dndy)
    both.add(PION_PDG, xf, mean_pt, np.full(len(xf), 0.01), observable="meanpT_xF",
             emulator=emulator)

    only_bulk = EmulatorLikelihood(bulk).add(PION_PDG, y, dndy, 0.05 * dndy)
    only_bs = EmulatorLikelihood(emulator).add(
        PION_PDG, xf, mean_pt, np.full(len(xf), 0.01), observable="meanpT_xF"
    )

    thetas = np.array([[0.0, 1.0, 0.4], [0.2, 0.9, 0.3], [-0.3, 1.1, 0.5]])
    np.testing.assert_allclose(both(thetas), only_bulk(thetas) + only_bs(thetas))
    assert np.argmax(both(thetas)) == 0

    other_paths = dict(PARAMETER_PATHS)
    other_paths["sigma"] = other_paths.pop("y_sigma")
    other = BulkEmulator(bulk_results, other_paths).fit()
    with pytest.raises(ValueError, match="differ"):
        EmulatorLikelihood(other).add(
            PION_PDG, xf, mean_pt, np.full(len(xf), 0.01), observable="meanpT_xF",
            emulator=emulator,
        )
