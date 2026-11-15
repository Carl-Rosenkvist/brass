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
from brass.testing import (
    analysis_args,
    concatenate_particles,
    generate_particles,
    write_run,
)

PION_PDG = 211
PION_MASS = 0.13957
PROTON_PDG = 2212
PROTON_MASS = 0.93827
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
        protons = generate_particles(
            np.random.default_rng(500 + i), 3000, pdg=PROTON_PDG,
            mass=PROTON_MASS, y_mean=y_mean, y_sigma=0.6, pt_mean=pt_mean,
        )
        with write_run(run, ["p0", "px", "py", "pz", "pdg"], config=config) as w:
            w.particles(concatenate_particles(pions, protons))
        out.append((run, pions))
    return out


@pytest.fixture(scope="module")
def results(runs):
    args = analysis_args(
        BaryonStopping, "--pdgs", str(PION_PDG), str(PROTON_PDG), "--xf-bins", "40"
    )
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
        emulator.axis(3312, "dNdxF")


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


# -- per-particle observables, x_F grouping and range ---------------------------


def test_xf_group_and_range(results):
    data = BaryonStoppingData(results, PARAMETER_PATHS)
    n = data.counts["xf"][PION_PDG]  # 40 bins of 0.05

    grouped = data.observable(PION_PDG, "dNdxF", xf_group=4)  # bins of 0.2
    np.testing.assert_allclose(grouped, n.reshape(len(n), 10, 4).sum(2) / 0.2)
    np.testing.assert_allclose(
        data.axis(PION_PDG, "dNdxF", xf_group=4), np.linspace(-0.9, 0.9, 10)
    )

    # Only bins entirely within |x_F| <= 0.2.
    np.testing.assert_allclose(
        data.axis(PION_PDG, "dNdxF", xf_group=2, xf_max=0.2), [-0.15, -0.05, 0.05, 0.15]
    )
    pt = data.counts["xf_pt"][PION_PDG]
    mean = data.observable(PION_PDG, "meanpT_xF", xf_group=2, xf_max=0.2)
    centre = slice(16, 24)
    expected = (pt[:, centre].reshape(-1, 4, 2).sum(2)
                / n[:, centre].reshape(-1, 4, 2).sum(2))
    np.testing.assert_allclose(mean, expected)

    with pytest.raises(ValueError, match="does not divide"):
        data.observable(PION_PDG, "dNdxF", xf_group=3)


def test_no_usable_bins_is_reported():
    data = BaryonStoppingData(fake_results([[3, 0], [0, 5]]), {"a": ("Meta", "a")})
    with pytest.raises(ValueError, match="PDG 211: no usable bins for 'meanpT_xF'"):
        data.observable(PION_PDG, "meanpT_xF")
    with pytest.raises(ValueError, match="no usable bins"):
        data.observable(PION_PDG, "dNdxF", xf_max=0.5)  # bins are [-1, 0] and [0, 1]

    emu = BaryonStoppingEmulator(
        fake_results([[3, 0], [0, 5]]), {"a": ("Meta", "a")}, observables=["meanpT_xF"]
    )
    with pytest.raises(ValueError, match="PDG 211"):
        emu.fit()


@pytest.fixture(scope="module")
def per_particle(results):
    return BaryonStoppingEmulator(
        results,
        PARAMETER_PATHS,
        observables={PION_PDG: ["dNdxF", "meanpT_xF"], PROTON_PDG: ["meanpT_xF"]},
        xf_max={PROTON_PDG: 0.2},
        xf_group=2,
    ).fit()


def test_per_particle_observables(per_particle):
    emu = per_particle
    assert emu.pdgs == [PION_PDG, PROTON_PDG]
    assert set(emu.models) == {
        (PION_PDG, "dNdxF"), (PION_PDG, "meanpT_xF"), (PROTON_PDG, "meanpT_xF")
    }
    np.testing.assert_allclose(emu.axis(PROTON_PDG, "meanpT_xF"), [-0.15, -0.05, 0.05, 0.15])
    assert np.allclose(np.diff(emu.axis(PION_PDG, "dNdxF")), 0.1)

    theta = [[0.0, 1.0, 0.4]]
    assert emu.predict_observable(theta, PROTON_PDG, "meanpT_xF").shape == (1, 4)
    with pytest.raises(ValueError, match="cannot be obtained"):
        emu.predict_observable(theta, PROTON_PDG, "dNdxF")

    # Only particles that have the observable are validated.
    results = emu.validate(holdout=0.25, observables=["dNdxF"])
    assert list(results) == [(PION_PDG, "dNdxF")]
    with pytest.raises(ValueError, match="cannot be obtained"):
        emu.validate(observables=["dNdy"])


def test_per_particle_settings_are_checked(results):
    with pytest.raises(KeyError, match="not in the data"):
        BaryonStoppingEmulator(results, PARAMETER_PATHS, observables={3312: ["dNdxF"]})
    with pytest.raises(KeyError, match="not emulated"):
        BaryonStoppingEmulator(
            results, PARAMETER_PATHS, observables={PION_PDG: ["dNdxF"]},
            xf_max={PROTON_PDG: 0.2},
        )


def test_per_particle_save_and_load(per_particle, tmp_path):
    path = tmp_path / "bs.npz"
    per_particle.save(path)
    loaded = BaryonStoppingEmulator.load(path)

    assert loaded.pdgs == per_particle.pdgs
    assert loaded.xf_max == per_particle.xf_max
    assert loaded.xf_group == per_particle.xf_group
    theta = [[0.1, 1.0, 0.4]]
    for pdg, name in per_particle.models:
        np.testing.assert_allclose(loaded.axis(pdg, name), per_particle.axis(pdg, name))
        np.testing.assert_allclose(
            loaded.predict_observable(theta, pdg, name),
            per_particle.predict_observable(theta, pdg, name),
            rtol=1e-10,
        )
    with pytest.raises(ValueError, match="cannot be obtained"):
        loaded.predict_observable(theta, PROTON_PDG, "dNdxF")



# -- folding and merging to a minimum count -----------------------------------------


def xf_results(counts, edges, pt=0.4, events=1):
    """BaryonStopping-like results: x_F counts per run (sqrt(s) = 2, so p_z = x_F)."""

    def histogram(values, e):
        return SimpleNamespace(values=np.asarray(values, float), edges=[np.asarray(e, float)])

    out = []
    for i, n in enumerate(counts):
        n = np.asarray(n, float)
        analysis = SimpleNamespace(
            config={"Meta": {"a": float(i)}}, events=events, folder_path=f"run{i}", sqrts=2.0,
            histograms={
                "xf": {PION_PDG: histogram(n, edges)},
                "xf_pt": {PION_PDG: histogram(pt * n, edges)},
                "y": {PION_PDG: histogram([1.0], [-1.0, 1.0])},
                "y_mt": {PION_PDG: histogram([0.5], [-1.0, 1.0])},
            },
        )
        out.append({"result": analysis})
    return out


A = {"a": ("Meta", "a")}
EDGES = np.linspace(-1, 1, 11)  # bins of 0.2
RIGHT = np.array([[5, 3, 2, 1, 0], [6, 1, 1, 2, 0]])  # inner -> outer, two runs
SYMMETRIC = np.hstack([RIGHT[:, ::-1], RIGHT])


def test_data_stores_events():
    data = BaryonStoppingData(xf_results(SYMMETRIC, EDGES, events=3), A)
    np.testing.assert_array_equal(data.events, [3, 3])


def test_fold_adds_both_sides(results):
    data = BaryonStoppingData(results, PARAMETER_PATHS)
    n = data.counts["xf"][PION_PDG]
    s = data.counts["xf_pt"][PION_PDG]
    n_f = n[:, 20:] + n[:, :20][:, ::-1]
    s_f = s[:, 20:] + s[:, :20][:, ::-1]

    np.testing.assert_allclose(data.axis(PION_PDG, "dNdxF", fold=True), np.arange(20) * 0.05 + 0.025)
    np.testing.assert_allclose(data.observable(PION_PDG, "dNdxF", fold=True), n_f / 2 / 0.05)
    filled = np.all(n_f > 0, axis=0)
    np.testing.assert_allclose(
        data.observable(PION_PDG, "meanpT_xF", fold=True), (s_f / np.where(n_f > 0, n_f, 1))[:, filled]
    )


def test_fold_odd_number_of_bins():
    edges = [-0.75, -0.25, 0.25, 0.75]
    data = BaryonStoppingData(xf_results([[1, 2, 3]], edges), A)
    lo, hi = data.bins(PION_PDG, "dNdxF", fold=True)
    np.testing.assert_allclose(lo, [0.0, 0.25])
    np.testing.assert_allclose(hi, [0.25, 0.75])
    # The middle bin keeps all its particles; dN/dx_F is the average of both sides.
    np.testing.assert_allclose(data.observable(PION_PDG, "dNdxF", fold=True), [[4.0, 4.0]])

    with pytest.raises(ValueError, match="symmetric"):
        BaryonStoppingData(xf_results([[1, 2]], [-0.5, 0.0, 1.0]), A).observable(
            PION_PDG, "dNdxF", fold=True
        )


def test_min_count_merges_outward():
    data = BaryonStoppingData(xf_results(SYMMETRIC, EDGES), A)

    # Without folding each side is merged separately: [0, 0.2] has 5 and 6
    # particles, then [0.2, 0.8] (3+2+1, 1+1+2); the empty outer bin is dropped.
    lo, hi = data.bins(PION_PDG, "dNdxF", min_count=4)
    np.testing.assert_allclose(lo, [-0.8, -0.2, 0.0, 0.2])
    np.testing.assert_allclose(hi, [-0.2, 0.0, 0.2, 0.8])
    np.testing.assert_allclose(
        data.observable(PION_PDG, "dNdxF", min_count=4),
        [[6 / 0.6, 5 / 0.2, 5 / 0.2, 6 / 0.6], [4 / 0.6, 6 / 0.2, 6 / 0.2, 4 / 0.6]],
    )

    # Folded: (10, 12), then (6, 2) + (4, 2) = (10, 4); the rest (2, 4) stays
    # below 4 particles in one run and joins its inner neighbour.
    lo, hi = data.bins(PION_PDG, "dNdxF", fold=True, min_count=4)
    np.testing.assert_allclose(lo, [0.0, 0.2])
    np.testing.assert_allclose(hi, [0.2, 0.8])
    np.testing.assert_allclose(data.observable(PION_PDG, "meanpT_xF", fold=True, min_count=4), 0.4)
    np.testing.assert_allclose(
        data.observable(PION_PDG, "dNdxF", fold=True, min_count=4),
        [[10 / 0.4, 12 / 1.2], [12 / 0.4, 8 / 1.2]],
    )


def test_min_count_uses_particle_numbers():
    # Counts are stored per event; the minimum refers to particles per run.
    data = BaryonStoppingData(xf_results(SYMMETRIC, EDGES, events=10), A)
    np.testing.assert_allclose(
        data.bins(PION_PDG, "dNdxF", fold=True, min_count=4)[0], [0.0, 0.2]
    )
    with pytest.raises(ValueError, match="positive"):
        data.observable(PION_PDG, "dNdxF", min_count=0)


def test_min_count_after_xf_max():
    data = BaryonStoppingData(xf_results(SYMMETRIC, EDGES), A)
    # The range is cut first: folded bins [0, 0.2] (10, 12) and [0.2, 0.4]
    # (6, 2). The second stays below 4 in one run and joins the first.
    lo, hi = data.bins(PION_PDG, "dNdxF", fold=True, xf_max=0.4, min_count=4)
    np.testing.assert_allclose(lo, [0.0])
    np.testing.assert_allclose(hi, [0.4])
    np.testing.assert_allclose(
        data.observable(PION_PDG, "dNdxF", fold=True, xf_max=0.4, min_count=4),
        [[16 / 0.8], [14 / 0.8]],
    )


@pytest.fixture(scope="module")
def merged(results):
    return BaryonStoppingEmulator(
        results,
        PARAMETER_PATHS,
        observables={PION_PDG: ["dNdxF", "meanpT_xF"], PROTON_PDG: ["meanpT_xF"]},
        fold=True,
        min_count={PION_PDG: 200, PROTON_PDG: 100},
    ).fit()


def test_emulator_fold_and_min_count(merged, results):
    data = BaryonStoppingData(results, PARAMETER_PATHS)
    for pdg, name in merged.models:
        lo, hi = merged.bins(pdg, name)
        assert np.all(lo >= 0)
        np.testing.assert_allclose(merged.axis(pdg, name), 0.5 * (lo + hi))
        expected = data.bins(pdg, name, fold=True, min_count=merged.min_count[pdg])
        np.testing.assert_allclose(lo, expected[0])
        np.testing.assert_allclose(hi, expected[1])
        # Every training run has at least min_count particles in every bin.
        n = data.observable(pdg, "dNdxF", fold=True, min_count=merged.min_count[pdg])
        width = 2 * (expected[1] - expected[0]) if name == "dNdxF" else None
        if width is not None:
            assert np.all(n * width * data.events[:, None] >= merged.min_count[pdg] - 1e-9)

    with pytest.raises(KeyError, match="not emulated"):
        BaryonStoppingEmulator(
            results, PARAMETER_PATHS, observables={PION_PDG: ["dNdxF"]}, fold={PROTON_PDG: True}
        )


def test_fold_and_min_count_survive_save_and_load(merged, tmp_path):
    path = tmp_path / "bs.npz"
    merged.save(path)
    loaded = BaryonStoppingEmulator.load(path)
    assert loaded.fold == merged.fold
    assert loaded.min_count == merged.min_count
    for pdg, name in merged.models:
        for a, b in zip(loaded.bins(pdg, name), merged.bins(pdg, name)):
            np.testing.assert_allclose(a, b)


def test_likelihood_with_custom_matrix(emulator):
    """A matrix replaces the interpolation, e.g. to extrapolate to x_F = 0."""
    axis = emulator.axis(PION_PDG, "meanpT_xF")
    theta = np.array([[0.0, 1.0, 0.4], [0.2, 0.9, 0.3]])
    model = emulator.predict_observable(theta, PION_PDG, "meanpT_xF")

    # Two points: linear extrapolation below the first bin centre, and the
    # value of bin 3.
    W = np.zeros((2, len(axis)))
    t = (axis[0] - 0.0) / (axis[1] - axis[0])
    W[0, 0], W[0, 1] = 1 + t, -t
    W[1, 3] = 1.0
    y = W @ model[0]
    like = EmulatorLikelihood(emulator, covariance="none").add(
        PION_PDG, [0.0, axis[3]], y, [0.01, 0.01], observable="meanpT_xF", matrix=W
    )
    assert like.n_points == 2  # x = 0 is kept although it is below the first centre
    chi2 = like.chi2(theta)
    assert chi2[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(chi2[1], np.sum(((W @ model[1]) - y) ** 2) / 0.01**2)

    with pytest.raises(ValueError, match="shape"):
        EmulatorLikelihood(emulator).add(
            PION_PDG, [0.0], [1.0], [0.1], observable="meanpT_xF", matrix=np.ones((1, 2))
        )
