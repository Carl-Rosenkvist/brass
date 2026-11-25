import numpy as np
import pytest

# Emulation needs the optional extra: pip install -e ".[emulation]"
pytest.importorskip("scipy")
pytest.importorskip("joblib")

from brass.emulation import PCAGP, Emulator, EmulatorLikelihood
from brass.emulation.gp import GaussianProcess, _negative_log_likelihood, optimise

# -- synthetic model ------------------------------------------------------------------

EDGES = np.linspace(-3, 3, 31)


def bumps(X):
    """Gaussian bumps whose height, centre and width depend on X (30 bins)."""
    x = 0.5 * (EDGES[1:] + EDGES[:-1])
    height, centre, width = 1 + X[:, 0], X[:, 1], 0.8 + 0.3 * X[:, 0]
    return height[:, None] * np.exp(
        -0.5 * ((x - centre[:, None]) / width[:, None]) ** 2
    )


def smooth(X):
    return np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2


@pytest.fixture(scope="module")
def design():
    return np.random.default_rng(1).uniform(0, 1, size=(60, 2))


@pytest.fixture(scope="module")
def emulator(design):
    emu = Emulator({"a": design[:, 0], "b": design[:, 1]}, n_components=0.999)
    emu.add("bumps", bumps(design), bins=EDGES)
    emu.add((7, "smooth"), smooth(design))
    return emu.fit(n_jobs=1)


# -- Gaussian process ---------------------------------------------------------------


def test_gradient_matches_finite_differences(design):
    from scipy.optimize import approx_fprime

    y = smooth(design)
    p = np.log([2.0, 0.7, 1.5, 3e-3])
    _, grad = _negative_log_likelihood(p, design, y, True)
    numeric = approx_fprime(
        p, lambda q: _negative_log_likelihood(q, design, y, True)[0], 1e-7
    )
    np.testing.assert_allclose(grad, numeric, rtol=1e-4, atol=1e-6)


def test_optimum_matches_sklearn(design):
    pytest.importorskip("sklearn")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    noise = 0.05 * np.random.default_rng(4).normal(size=len(design))
    y, _, _ = GaussianProcess.standardise(smooth(design) + noise)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        np.ones(2), (1e-2, 1e3), nu=2.5
    ) + WhiteKernel(1e-2, (1e-8, 1e1))
    reference = GaussianProcessRegressor(kernel, alpha=1e-10).fit(design, y)

    hyper = optimise(design, y)
    p = np.log([hyper["amplitude"], *hyper["length_scale"], hyper["noise_level"]])
    ours = -_negative_log_likelihood(p, design, y, True)[0]
    np.testing.assert_allclose(
        ours, reference.log_marginal_likelihood_value_, rtol=1e-6
    )


def test_parallel_fit_is_the_same(design):
    serial = PCAGP().fit(design, bumps(design), n_jobs=1)
    parallel = PCAGP().fit(design, bumps(design), n_jobs=2)
    for a, b in zip(serial.gps, parallel.gps):
        np.testing.assert_allclose(a.length_scale, b.length_scale)
        assert a.noise_level == b.noise_level


# -- Emulator -------------------------------------------------------------------------


def test_predicts_held_out_points(emulator):
    X = np.random.default_rng(3).uniform(0.1, 0.9, size=(20, 2))
    mean, std = emulator.predict(X, "bumps", return_std=True)
    assert mean.shape == std.shape == (20, 30)
    assert np.max(np.abs(mean - bumps(X))) < 0.05
    assert np.max(np.abs(emulator.predict(X, (7, "smooth"))[:, 0] - smooth(X))) < 0.05


def test_parameters_by_name(emulator):
    X = np.array([[0.3, 0.6], [0.5, 0.6]])
    by_name = emulator.predict({"a": [0.3, 0.5], "b": 0.6}, "bumps")
    np.testing.assert_allclose(by_name, emulator.predict(X, "bumps"))
    np.testing.assert_allclose(
        emulator.predict({"b": 0.6, "a": 0.3}, "bumps"),
        emulator.predict(X[:1], "bumps"),
    )
    with pytest.raises(KeyError, match="missing"):
        emulator.predict({"a": 0.3}, "bumps")
    with pytest.raises(ValueError, match="columns"):
        emulator.predict([0.1, 0.2, 0.3], "bumps")


def test_all_outputs_std_cov_and_linear_map(emulator):
    X = np.array([[0.4, 0.5]])
    everything = emulator.predict(X, return_std=True)
    assert list(everything) == ["bumps", (7, "smooth")]

    mean, cov = emulator.predict(X, "bumps", return_cov=True)
    np.testing.assert_allclose(mean, everything["bumps"][0])
    np.testing.assert_allclose(np.sqrt(np.diagonal(cov[0])), everything["bumps"][1][0])

    A = np.diff(EDGES)[None, :]  # integral over all bins
    total, total_std = emulator.predict(X, "bumps", A, return_std=True)
    np.testing.assert_allclose(total[:, 0], mean[0] @ A[0])
    np.testing.assert_allclose(total_std[0, 0], np.sqrt(A[0] @ cov[0] @ A[0]))


def test_axis_and_bins(emulator):
    np.testing.assert_allclose(emulator.axis("bumps"), 0.5 * (EDGES[1:] + EDGES[:-1]))
    lo, hi = emulator.bins("bumps")
    np.testing.assert_allclose(hi - lo, np.diff(EDGES))
    assert emulator.axis((7, "smooth")) is None
    with pytest.raises(KeyError, match="no output"):
        emulator.axis("other")


def test_save_and_load(emulator, design, tmp_path):
    emulator.save(tmp_path / "emu.npz")
    loaded = Emulator.load(tmp_path / "emu.npz")
    assert loaded.keys == emulator.keys
    assert loaded.parameter_names == ["a", "b"]
    for key in emulator.keys:
        np.testing.assert_allclose(
            loaded.predict(design[:5], key, return_std=True),
            emulator.predict(design[:5], key, return_std=True),
        )
    np.testing.assert_allclose(loaded.bins("bumps"), emulator.bins("bumps"))
    with pytest.raises(RuntimeError, match="no training data"):
        loaded.fit()


def test_validate_holds_out_whole_points():
    rng = np.random.default_rng(5)
    points = rng.uniform(0, 1, size=(40, 2))
    energy = np.repeat([1.0, 2.0], 40)  # every point at both energies
    X = np.column_stack([energy, np.vstack([points, points])])
    emu = Emulator(X, names=["E", "a", "b"])
    emu.add("y", np.column_stack([smooth(X[:, 1:]) * X[:, 0], X[:, 1] + X[:, 0]]))
    results = emu.fit(n_jobs=1).validate(holdout=0.25, n_jobs=1)

    r = results["y"]
    held = {tuple(p) for p in r["parameters"][:, 1:]}
    assert len(held) == 10 and len(r["parameters"]) == 20  # both energies of 10 points
    assert np.median(np.abs(r["mean"] - r["truth"])) < 0.05
    assert 0.3 < np.std(r["residuals"]) < 3


def test_errors(design):
    emu = Emulator({"a": design[:, 0], "b": design[:, 1]})
    emu.add(1, smooth(design))
    with pytest.raises(ValueError, match="already added"):
        emu.add(1, smooth(design))
    with pytest.raises(ValueError, match="one row per run"):
        emu.add(2, smooth(design)[:10])
    with pytest.raises(ValueError, match="finite"):
        emu.add(3, np.full(len(design), np.nan))
    with pytest.raises(TypeError, match="keys"):
        emu.add(1.5, smooth(design))
    with pytest.raises(ValueError, match="edges"):
        emu.add(4, bumps(design), bins=EDGES[:-1])
    with pytest.raises(RuntimeError, match="not fitted"):
        emu.predict(design[:1], 1)
    with pytest.raises(ValueError, match="one value per run"):
        Emulator({"a": [1, 2], "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="log_parameters"):
        Emulator({"a": [1, 2]}, log_parameters=["c"])


# -- likelihood -----------------------------------------------------------------------


def test_likelihood_matches_manual(emulator):
    x = np.linspace(-2, 2, 7)
    theta = np.array([[0.4, 0.5], [0.7, 0.2]])
    truth = bumps(theta[:1])[0]
    centres = emulator.axis("bumps")
    y = np.interp(x, centres, truth)
    yerr = 0.05 * y + 0.01

    like = EmulatorLikelihood(emulator).add("bumps", x, y, yerr)
    W = np.array([np.interp(x, centres, np.eye(len(centres))[j]) for j in range(30)]).T
    expected = []
    for t in theta:
        mean, cov = emulator.predict(t, "bumps", return_cov=True)
        C = np.diag(yerr**2) + W @ cov[0] @ W.T
        r = W @ mean[0] - y
        expected.append(
            -0.5
            * (
                r @ np.linalg.solve(C, r)
                + np.linalg.slogdet(C)[1]
                + 7 * np.log(2 * np.pi)
            )
        )
    np.testing.assert_allclose(like(theta), expected, rtol=1e-8)
    assert like.datasets[0]["label"].startswith("bumps")


def test_likelihood_fixed_parameter_and_scalar_output(emulator):
    like = EmulatorLikelihood(emulator)
    value = smooth(np.array([[0.3, 0.8]]))
    like.add((7, "smooth"), [0.0], value, [0.01], fixed={"a": 0.3})
    assert like.parameters == ["b"]
    logl = like(np.array([[0.8], [0.2]]))
    assert logl[0] > logl[1]


def test_likelihood_needs_bins_or_matrix(emulator):
    emu = Emulator(emulator.parameters, names=["a", "b"])
    emu.add("nobins", emulator._training["bumps"]).fit(n_jobs=1)
    with pytest.raises(ValueError, match="no bins"):
        EmulatorLikelihood(emu).add("nobins", [0.0], [1.0], [0.1])
    with pytest.raises(TypeError, match="observable"):
        EmulatorLikelihood(emulator).add(
            "bumps", [0.0], [1.0], [0.1], observable="dNdy"
        )
    W = np.zeros((1, 30))
    W[0, 15] = 1.0
    EmulatorLikelihood(emu).add("nobins", [0.0], [1.0], [0.1], matrix=W)
