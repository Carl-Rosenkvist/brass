import numpy as np
import pytest
from brass import AnalysisRunner
from brass.analyses import BaryonStopping
from brass.testing import (
    analysis_args,
    concatenate_particles,
    generate_particles,
    write_run,
)

PROTON_PDG = 2212
LAMBDA_PDG = 3122
SIGMA0_PDG = 3212
PROTON_MASS = 0.93827
LAMBDA_MASS = 1.11568

SQRTS = 8.76
QUANTITIES = ["p0", "px", "py", "pz", "pdg"]
ARGS = ("--xf-bins", "20", "--y-bins", "16")


def args(*pdgs, extra=()):
    return analysis_args(BaryonStopping, "--pdgs", *map(str, pdgs), *ARGS, *extra)


def make_run(folder, seed, sqrts=SQRTS, frame="center of mass", **collider):
    rng = np.random.default_rng(seed)
    protons = generate_particles(
        rng, 20_000, pdg=PROTON_PDG, mass=PROTON_MASS,
        y_mean=0.3, y_sigma=0.8, pt_mean=0.4,
    )
    lambdas = generate_particles(rng, 2_000, pdg=LAMBDA_PDG, mass=LAMBDA_MASS)
    sigmas = generate_particles(rng, 1_000, pdg=SIGMA0_PDG, mass=1.19264)
    config = {
        "Modi": {"Collider": {"Sqrtsnn": sqrts, "Calculation_Frame": frame, **collider}}
    }
    with write_run(folder, QUANTITIES, config=config) as w:
        w.particles(concatenate_particles(protons, lambdas, sigmas))
    return {PROTON_PDG: protons, LAMBDA_PDG: lambdas, SIGMA0_PDG: sigmas}


def values(histogram):
    return np.array(histogram.values)


def test_histograms_are_exact(tmp_path):
    particles = make_run(tmp_path / "run", seed=1)[PROTON_PDG]
    analysis = BaryonStopping(tmp_path / "run", args(PROTON_PDG))
    analysis.run()

    assert analysis.sqrts == SQRTS
    assert analysis.events == 1
    h = {kind: hist[PROTON_PDG] for kind, hist in analysis.histograms.items()}

    xf = 2 * particles["pz"] / SQRTS
    xf_edges = np.linspace(-1, 1, 21)
    np.testing.assert_allclose(2 * np.asarray(h["xf"].edges[0]) / SQRTS, xf_edges)
    np.testing.assert_array_equal(values(h["xf"]), np.histogram(xf, xf_edges)[0])
    np.testing.assert_allclose(
        values(h["xf_pt"]), np.histogram(xf, xf_edges, weights=particles["pt"])[0]
    )

    y_edges = np.linspace(-4, 4, 17)
    np.testing.assert_allclose(np.asarray(h["y"].edges[0]), y_edges)
    np.testing.assert_array_equal(values(h["y"]), np.histogram(particles["y"], y_edges)[0])
    np.testing.assert_allclose(
        values(h["y_mt"]), np.histogram(particles["y"], y_edges, weights=particles["mt"])[0]
    )


def test_add_pdgs(tmp_path):
    particles = make_run(tmp_path / "run", seed=2)
    analysis = BaryonStopping(
        tmp_path / "run",
        args(LAMBDA_PDG, extra=("--add-pdgs", str(LAMBDA_PDG), str(SIGMA0_PDG))),
    )
    analysis.run()

    assert set(analysis.histograms["xf"]) == {LAMBDA_PDG}
    both = concatenate_particles(particles[LAMBDA_PDG], particles[SIGMA0_PDG])
    expected, _ = np.histogram(both["y"], np.linspace(-4, 4, 17))
    np.testing.assert_array_equal(values(analysis.histograms["y"][LAMBDA_PDG]), expected)


def test_merge(tmp_path):
    analyses = []
    for i in range(2):
        make_run(tmp_path / f"run{i}", seed=10 + i)
        analyses.append(BaryonStopping(tmp_path / f"run{i}", args(PROTON_PDG)))
        analyses[-1].run()
    single = [values(a.histograms["xf_pt"][PROTON_PDG]) for a in analyses]

    analyses[0] += analyses[1]
    np.testing.assert_allclose(
        values(analyses[0].histograms["xf_pt"][PROTON_PDG]), single[0] + single[1]
    )
    assert analyses[0].events == 2

    make_run(tmp_path / "other", seed=5, sqrts=17.3)
    other = BaryonStopping(tmp_path / "other", args(PROTON_PDG))
    other.run()
    with pytest.raises(ValueError, match="different energies"):
        analyses[0] += other


def test_runner_groups_runs_by_energy(tmp_path):
    folders = []
    for i, sqrts in enumerate((SQRTS, SQRTS, 17.3)):
        make_run(tmp_path / f"run{i}", seed=20 + i, sqrts=sqrts)
        folders.append(tmp_path / f"run{i}")
    results = AnalysisRunner(BaryonStopping, folders, args=args(PROTON_PDG)).run()

    events = sorted((r["result"].sqrts, r["result"].events) for r in results)
    assert events == [(SQRTS, 2), (17.3, 1)]


def test_needs_centre_of_mass_collider(tmp_path):
    make_run(tmp_path / "fixed", seed=1, frame="fixed target")
    with pytest.raises(ValueError, match="centre-of-mass"):
        BaryonStopping(tmp_path / "fixed", args(PROTON_PDG))

    make_run(
        tmp_path / "asym", seed=1, frame="center of velocity",
        Projectile={"Particles": {2212: 1}}, Target={"Particles": {2212: 1, 2112: 1}},
    )
    with pytest.raises(ValueError, match="centre-of-mass"):
        BaryonStopping(tmp_path / "asym", args(PROTON_PDG))

    # Symmetric systems are fine in the centre-of-velocity frame.
    make_run(
        tmp_path / "sym", seed=1, frame="center of velocity",
        Projectile={"Particles": {2212: 1}}, Target={"Particles": {2212: 1}},
    )
    BaryonStopping(tmp_path / "sym", args(PROTON_PDG))

    with write_run(tmp_path / "nosqrts", QUANTITIES, config={"Meta": {}}) as w:
        w.particles([[1.0, 0.1, 0.1, 0.1, PROTON_PDG]])
    with pytest.raises(ValueError, match="Sqrtsnn"):
        BaryonStopping(tmp_path / "nosqrts", args(PROTON_PDG))


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--xf-bins", "0"), "--xf-bins"),
        (("--y-bins", "0"), "--y-bins"),
        (("--y-range", "1", "1"), "--y-range"),
        (("--add-pdgs", "3122"), "at least two"),
        (("--add-pdgs", "3212", "3122"), "must be included"),
    ],
)
def test_rejects_bad_arguments(tmp_path, extra, message):
    make_run(tmp_path / "run", seed=1)
    with pytest.raises(ValueError, match=message):
        BaryonStopping(tmp_path / "run", args(PROTON_PDG, LAMBDA_PDG, extra=extra))
