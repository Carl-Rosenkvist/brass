import numpy as np
import pytest

pytest.importorskip("sklearn")

from brass.emulation import PCAGP
from brass.emulation.gp import GaussianProcess


def smooth(X):
    return np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2


def spectra(X, x=np.linspace(-3, 3, 30)):
    """Gaussian bumps whose height, centre and width depend on X."""
    height = 1 + X[:, 0]
    centre = X[:, 1]
    width = 0.8 + 0.3 * X[:, 0]
    return height[:, None] * np.exp(
        -0.5 * ((x[None, :] - centre[:, None]) / width[:, None]) ** 2
    )


@pytest.fixture
def design():
    rng = np.random.default_rng(1)
    return rng.uniform(0, 1, size=(60, 2))


def test_gp_matches_sklearn(design):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    y = smooth(design)
    gp = GaussianProcess(noise=True).fit(design, y)

    kernel = ConstantKernel(gp.amplitude) * Matern(gp.length_scale, nu=2.5)
    reference = GaussianProcessRegressor(
        kernel + WhiteKernel(gp.noise_level), optimizer=None, alpha=1e-10
    ).fit(design, (y - gp.y_mean) / gp.y_std)

    X = np.random.default_rng(2).uniform(0, 1, size=(10, 2))
    mean, var = gp.predict(X, return_var=True)
    ref_mean, ref_std = reference.predict(X, return_std=True)

    np.testing.assert_allclose(mean, gp.y_mean + gp.y_std * ref_mean, rtol=1e-8)
    # sklearn's std includes the white noise; ours is the latent function's.
    np.testing.assert_allclose(
        var, gp.y_std**2 * (ref_std**2 - gp.noise_level), rtol=1e-6, atol=1e-12
    )


def test_pcagp_predicts_held_out(design):
    model = PCAGP(n_components=0.999).fit(design, spectra(design))

    X = np.random.default_rng(3).uniform(0.1, 0.9, size=(20, 2))
    mean, std = model.predict(X, return_std=True)

    truth = spectra(X)
    assert np.max(np.abs(mean - truth)) < 0.05
    assert np.all(std > 0)


def test_pcagp_std_cov_and_linear_map_agree(design):
    model = PCAGP(n_components=2).fit(design, spectra(design))
    X = design[:5] + 0.01

    mean, std = model.predict(X, return_std=True)
    mean_c, cov = model.predict(X, return_cov=True)
    np.testing.assert_allclose(mean, mean_c)
    np.testing.assert_allclose(std**2, np.diagonal(cov, axis1=1, axis2=2))

    A = np.random.default_rng(4).normal(size=(3, mean.shape[1]))
    mean_a, cov_a = model.predict(X, A, return_cov=True)
    np.testing.assert_allclose(mean_a, mean @ A.T)
    np.testing.assert_allclose(cov_a, A @ cov @ A.T, atol=1e-12)


def test_truncation_error_is_included(design):
    Y = spectra(design)
    truncated = PCAGP(n_components=1).fit(design, Y)
    without = PCAGP(n_components=1, include_truncation=False).fit(design, Y)

    assert truncated.explained_fraction < 1
    _, std_with = truncated.predict(design[:3], return_std=True)
    _, std_without = without.predict(design[:3], return_std=True)
    assert np.all(std_with > std_without)


def test_pcagp_state_round_trip(design):
    model = PCAGP(n_components=0.99, log_parameters=[0]).fit(
        design + 0.5, spectra(design)
    )
    loaded = PCAGP.from_state(model.state())

    X = design[:7] + 0.55
    for a, b in zip(
        model.predict(X, return_cov=True), loaded.predict(X, return_cov=True)
    ):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)


def test_log_parameters_must_be_positive(design):
    with pytest.raises(ValueError, match="positive"):
        PCAGP(log_parameters=[0]).fit(design - 1, spectra(design))
