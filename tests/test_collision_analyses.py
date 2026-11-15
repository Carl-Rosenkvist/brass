"""Analyses of SMASH collision (interaction-block) output."""

import numpy as np
import pytest

from brass import AnalysisRunner
from brass.analyses import CollisionCounts
from brass.testing import analysis_args, write_run

QUANTITIES = ["t", "p0", "px", "py", "pz", "pdg"]
PI0 = 111


def particles(t, *rows):
    """Rows (pdg, px, py, pz, mass) at time t as a column mapping."""
    rows = np.array(rows, dtype=float).reshape(-1, 5)
    pdg, px, py, pz, mass = rows.T
    return {
        "t": np.full(len(rows), t),
        "p0": np.sqrt(mass**2 + px**2 + py**2 + pz**2),
        "px": px,
        "py": py,
        "pz": pz,
        "pdg": pdg.astype(np.int32),
    }


def elastic(writer, t, process=1):
    pair = particles(t, (PI0, 0.1, 0, 0.2, 0.138), (PI0, -0.1, 0, -0.2, 0.138))
    writer.interaction(pair, pair, process=process, sigma=10.0)


def test_collision_counts(tmp_path):
    with write_run(tmp_path / "run", QUANTITIES, output="Collisions",
                   config={"General": {"End_Time": 60.0}}) as w:
        for event in range(3):
            elastic(w, 1.0)  # before t_start
            elastic(w, 7.0)
            elastic(w, 8.0)
            decay = particles(9.0, (113, 0, 0, 0, 0.775))
            pions = particles(9.0, (211, 0.3, 0, 0, 0.138), (-211, -0.3, 0, 0, 0.138))
            w.interaction(decay, pions, process=5)
            w.end(event=event)

    args = analysis_args(CollisionCounts, "--t-start", "5")
    analysis = CollisionCounts(tmp_path / "run", args)
    analysis.run()

    assert analysis.events == 3
    assert analysis.counts_for(process=1, n_in=2, n_out=2) == 6
    assert analysis.counts_for(process=5) == 3
    assert analysis.counts_for(n_in=1) == 3
    assert analysis.counts_for() == 9


def test_collision_counts_merge(tmp_path):
    for name in ("a", "b"):
        with write_run(tmp_path / name, QUANTITIES, output="Collisions") as w:
            elastic(w, 2.0)
            w.end(event=0)
    args = analysis_args(CollisionCounts)
    results = AnalysisRunner(CollisionCounts, [tmp_path / "a", tmp_path / "b"], args=args, max_workers=1).run()
    merged = results[0]["result"]
    assert merged.events == 2 and merged.counts_for(process=1) == 2

    other = CollisionCounts(tmp_path / "a", analysis_args(CollisionCounts, "--t-start", "1"))
    with pytest.raises(ValueError, match="t-start"):
        merged += other


# --- detailed balance -----------------------------------------------------

from brass.analyses import MultiplicityVsTime, ReactionCounts  # noqa: E402

RHO0, PIP, PIM = 113, 211, -211


def pion_pair(t, p):
    return particles(t, (PIP, 0, 0, p, 0.138), (PIM, 0, 0, -p, 0.138))


def write_reactions(folder):
    """Per event: rho formation, rho decay, one early formation, one wall crossing."""
    with write_run(folder, QUANTITIES, output="Collisions",
                   config={"General": {"End_Time": 100.0}}) as w:
        for event in range(2):
            pions = pion_pair(30.0, 0.35)  # sqrt(s) = 2 sqrt(0.35^2 + 0.138^2) = 0.7524
            rho = particles(30.0, (RHO0, 0, 0, 0, 0.7524))
            w.interaction(pions, rho, process=2)
            w.interaction(rho, pion_pair(31.0, 0.35), process=5)
            w.interaction(pion_pair(5.0, 0.35), rho, process=2)  # before t_start
            wall = particles(40.0, (PIP, 0.1, 0, 0, 0.138))
            w.interaction(wall, wall, process=6)
            w.end(event=event)


def test_reaction_counts(tmp_path):
    write_reactions(tmp_path / "run")
    args = analysis_args(ReactionCounts, "--t-start", "20")
    analysis = ReactionCounts(tmp_path / "run", args)
    analysis.run()

    formation = ((PIM, PIP), (RHO0,))
    decay = analysis.inverse(formation)
    assert set(analysis.counts) == {formation, decay}
    assert analysis.events == 2

    sqrts = 2 * np.hypot(0.35, 0.138)
    bin_ = int(sqrts / 0.05)
    assert analysis.histogram(formation)[bin_] == 2
    assert analysis.histogram(formation).sum() == 2
    assert analysis.histogram(decay)[bin_] == 2
    assert analysis.pairs() == [(formation, decay)] or analysis.pairs() == [(decay, formation)]
    rate = analysis.rate(formation)
    assert rate[bin_] == pytest.approx(2 / (2 * 80.0 * 0.05))
    assert analysis.histogram(((1,), (2,))).sum() == 0
    assert len(analysis.most_violating(6)) == 1


def test_reaction_counts_t(tmp_path):
    with write_run(tmp_path / "run", QUANTITIES, output="Collisions") as w:
        incoming = particles(1.0, (PI0, 0, 0, 0.5, 0.138), (PI0, 0, 0, -0.5, 0.138))
        outgoing = particles(1.0, (PI0, 0.5, 0, 0, 0.138), (PI0, -0.5, 0, 0, 0.138))
        w.interaction(incoming, outgoing, process=1)
        w.interaction(particles(1.0, (RHO0, 0, 0, 0, 0.77)), pion_pair(1.0, 0.3), process=5)
        w.end(event=0)
    analysis = ReactionCounts(tmp_path / "run", analysis_args(ReactionCounts, "--variable", "t"))
    analysis.run()
    # |t| = |(p_in - p_out)^2| = 0.5^2 + 0.5^2 = 0.5 GeV^2 -> bin [0.5, 1.0); the decay is not 2 -> 2
    assert list(analysis.counts) == [((PI0, PI0), (PI0, PI0))]
    assert analysis.histogram(((PI0, PI0), (PI0, PI0)))[1] == 1


def test_reaction_counts_merge(tmp_path):
    write_reactions(tmp_path / "a")
    write_reactions(tmp_path / "b")
    args = analysis_args(ReactionCounts, "--t-start", "20")
    results = AnalysisRunner(ReactionCounts, [tmp_path / "a", tmp_path / "b"], args=args, max_workers=1).run()
    merged = results[0]["result"]
    assert merged.events == 4
    assert merged.histogram(((PIM, PIP), (RHO0,))).sum() == 4


def test_multiplicity_vs_time(tmp_path):
    with write_run(tmp_path / "run", ["t", "p0", "px", "py", "pz", "pdg"]) as w:
        # event 0: 2, 1, 0 pi+ at t = 0, 10, 20; event 1: 4, 1, 2
        for event, numbers in enumerate([(2, 1, 0), (4, 1, 2)]):
            for t, n in zip((0.0, 10.0, 20.0), numbers):
                rows = [(PIP, 0, 0, 0, 0.138)] * n + [(PI0, 0, 0, 0, 0.138)]
                w.particles(particles(t, *rows), event=event)
            w.end(event=event)
    args = analysis_args(MultiplicityVsTime, "--pdgs", str(PIP), str(PI0))
    analysis = MultiplicityVsTime(tmp_path / "run", args)
    analysis.run()

    np.testing.assert_allclose(analysis.times, [0, 10, 20])
    assert analysis.events == 2
    np.testing.assert_allclose(analysis.mean(PIP), [3, 1, 1])
    np.testing.assert_allclose(analysis.mean(PI0), [1, 1, 1])
    # sample std / sqrt(n): event counts (2, 4) -> std 1.414, error 1
    np.testing.assert_allclose(analysis.error(PIP), [1, 0, 1])


# --- angular distributions ------------------------------------------------

from brass.analyses import AngularDistributions  # noqa: E402

PROTON, DELTA_PP, NEUTRON = 2212, 2224, 2112
M_P = 0.938


def nn_scattering(writer, theta, p=0.5, out=(PROTON, PROTON), masses=(M_P, M_P), sigma=40.0, process=1):
    """pp -> out in the centre-of-mass frame; the first outgoing particle at angle theta."""
    incoming = particles(1.0, (PROTON, 0, 0, p, M_P), (PROTON, 0, 0, -p, M_P))
    px, pz = p * np.sin(theta), p * np.cos(theta)
    outgoing = particles(1.0, (out[0], px, 0, pz, masses[0]), (out[1], -px, 0, -pz, masses[1]))
    writer.interaction(incoming, outgoing, sigma=sigma, process=process)


def test_angular_distributions(tmp_path):
    with write_run(tmp_path / "run", QUANTITIES, output="Collisions") as w:
        nn_scattering(w, 0.05)                        # elastic, first theta bin
        nn_scattering(w, 0.05)
        nn_scattering(w, 1.6)                         # elastic, bin 15
        nn_scattering(w, 0.05, out=(NEUTRON, DELTA_PP), masses=(0.938, 1.232), process=3)
        three = particles(1.0, (PROTON, 0, 0, 0.1, M_P), (PROTON, 0, 0, -0.1, M_P), (111, 0, 0, 0, 0.138))
        w.interaction(particles(1.0, (PROTON, 0, 0, 0.5, M_P), (PROTON, 0, 0, -0.5, M_P)), three, sigma=40.0)
        pion_p = particles(1.0, (211, 0, 0, 0.3, 0.138), (PROTON, 0, 0, -0.3, M_P))
        w.interaction(pion_p, pion_p, sigma=100.0, process=1)  # other incoming pair: ignored
        w.end(event=0)

    args = analysis_args(AngularDistributions, "--incoming", "2212", "2212")
    analysis = AngularDistributions(tmp_path / "run", args)
    analysis.run()

    assert analysis.total == 5  # four 2 -> 2 and one 2 -> 3
    assert analysis.channels() == [(NEUTRON, DELTA_PP), (PROTON, PROTON)]
    assert analysis.sigma == pytest.approx(40.0)

    dtheta = np.pi / 30
    theta, elastic = analysis.dsigma_dtheta([(PROTON, PROTON)])
    assert elastic[0] == pytest.approx(2 * 40.0 / (5 * dtheta))
    assert elastic[15] == pytest.approx(40.0 / (5 * dtheta))
    _, everything = analysis.dsigma_dtheta()
    assert everything[0] == pytest.approx(3 * 40.0 / (5 * dtheta))

    # elastic t_max = 4 p_cm^2 = 1; |t| of theta = 0.05 is 2 p^2 (1 - cos theta)
    assert analysis.t_edges[-1] == pytest.approx(1.0)
    t, dsdt = analysis.dsigma_dt([(PROTON, PROTON)])
    t_small = 2 * 0.25 * (1 - np.cos(0.05))
    assert dsdt[int(t_small / 0.02)] == pytest.approx(2 * 40.0 / (5 * 0.02))
    # integrating over theta gives the 2 -> 2 share of sigma
    assert (everything * dtheta).sum() == pytest.approx(4 / 5 * 40.0)


def test_angular_distributions_merge(tmp_path):
    for name in ("a", "b"):
        with write_run(tmp_path / name, QUANTITIES, output="Collisions") as w:
            nn_scattering(w, 0.5)
            w.end(event=0)
    args = analysis_args(AngularDistributions, "--incoming", "2212", "2212")
    results = AnalysisRunner(AngularDistributions, [tmp_path / "a", tmp_path / "b"], args=args, max_workers=1).run()
    merged = results[0]["result"]
    assert merged.total == 2 and merged.events == 2
    assert merged.theta_counts[(PROTON, PROTON)].sum() == 2
