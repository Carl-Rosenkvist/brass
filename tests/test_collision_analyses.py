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


# --- cross sections -------------------------------------------------------

from brass.analyses import CrossSections  # noqa: E402
from brass.analyses.cross_sections import transverse_distance  # noqa: E402

XS_QUANTITIES = ["t", "x", "y", "z", "p0", "px", "py", "pz", "pdg", "id"]
PI0_PDG = 111


def xs_particles(*rows):
    """Rows (t, x, y, z, pdg, px, py, pz, mass, id)."""
    rows = np.array(rows, dtype=float).reshape(-1, 10)
    t, x, y, z, pdg, px, py, pz, mass, ids = rows.T
    return {
        "t": t, "x": x, "y": y, "z": z,
        "p0": np.sqrt(mass**2 + px**2 + py**2 + pz**2), "px": px, "py": py, "pz": pz,
        "pdg": pdg.astype(np.int32), "id": ids.astype(np.int32),
    }


def pp_initial(b, p=1.0):
    return xs_particles((0, b / 2, 0, -2, PROTON, 0, 0, p, M_P, 0), (0, -b / 2, 0, 2, PROTON, 0, 0, -p, M_P, 1))


def write_xs_run(folder, events):
    """events: list of (b, process or None, final-state pdgs)."""
    config = {"Modi": {"Collider": {"Sqrtsnn": 3.0, "Impact": {"Max": 2.0}}}}
    with write_run(folder, XS_QUANTITIES, output="Collisions", config=config) as w:
        for n, (b, process, final) in enumerate(events):
            initial = pp_initial(b)
            w.particles(initial, event=n)
            if process is not None:
                out = xs_particles(*[(1, 0, 0, 0, pdg, 0.1 * i, 0, 0.5, 1.0, 2 + i) for i, pdg in enumerate(final)])
                w.interaction(initial, out, process=process, sigma=31.4)
                w.particles(out, event=n)
            else:
                w.particles(initial, event=n)
            w.end(event=n)


def test_transverse_distance_is_frame_independent():
    b, p, beta = 0.7, 1.0, 0.6
    e = np.hypot(M_P, p)
    x1, p1 = np.array([0, b / 2, 0, -2.0]), np.array([e, 0, 0, p])
    x2, p2 = np.array([0, -b / 2, 0, 2.0]), np.array([e, 0, 0, -p])
    assert transverse_distance(x1, p1, x2, p2) == pytest.approx(b)

    gamma = 1 / np.sqrt(1 - beta**2)

    def boost(v):  # along z into a frame moving with -beta
        return np.array([gamma * (v[0] + beta * v[3]), v[1], v[2], gamma * (v[3] + beta * v[0])])

    assert transverse_distance(boost(x1), boost(p1), boost(x2), boost(p2)) == pytest.approx(b)


def test_cross_sections(tmp_path):
    # black disc of radius 1 fm on a grid of impact parameters in [0, 2) fm
    bs = (np.arange(400) + 0.5) * 2.0 / 400
    events = []
    for i, b in enumerate(bs):
        if b >= 1.0:
            events.append((b, None, ()))
        elif i % 4 == 0:
            events.append((b, 3, (NEUTRON, DELTA_PP)))                      # inelastic
        elif i % 4 == 1:
            events.append((b, 45, (PROTON, PROTON)))                         # string back to pp
        else:
            events.append((b, 1, (PROTON, PROTON)))                          # elastic
    write_xs_run(tmp_path / "run", events)

    args = analysis_args(CrossSections, "--d-bins", "200")
    analysis = CrossSections(tmp_path / "run", args)
    analysis.run()

    assert analysis.n_events == 400 and analysis.initial == (PROTON, PROTON)
    sigma, error = analysis.total()
    assert sigma == pytest.approx(10 * np.pi * 1.0**2)  # P = 1 inside, 0 outside
    assert error == pytest.approx(0.0)

    elastic, _ = analysis.cross_section(analysis.is_elastic)
    assert elastic == pytest.approx(sigma * 100 / 200)
    finals = analysis.final_states()
    assert finals[(PROTON, PROTON)][0] == pytest.approx(sigma * 150 / 200)
    assert finals[(NEUTRON, DELTA_PP)][0] == pytest.approx(sigma * 50 / 200)
    assert analysis.first_channels()[(NEUTRON, DELTA_PP)][0] == pytest.approx(sigma / 4)
    # a weight per event: protons in the final state
    assert analysis.cross_section(lambda p, f, final: final.count(PROTON))[0] == pytest.approx(sigma * 300 / 200)
    types = analysis.process_types()
    assert types["elastic"][0] == pytest.approx(sigma / 2)
    assert types["resonance"][0] == pytest.approx(sigma / 4)
    assert types["soft-string"][0] == pytest.approx(sigma / 4)


def test_cross_sections_merge_and_partial_probability(tmp_path):
    # half of the events at b < 1 interact: sigma = pi / 2 fm^2
    bs = (np.arange(200) + 0.5) * 1.0 / 200
    write_xs_run(tmp_path / "a", [(b, 1 if i % 2 else None, (PROTON, PROTON)) for i, b in enumerate(bs)])
    write_xs_run(tmp_path / "b", [(b, 1 if i % 2 else None, (PROTON, PROTON)) for i, b in enumerate(bs)])
    args = analysis_args(CrossSections, "--d-bins", "4")
    results = AnalysisRunner(CrossSections, [tmp_path / "a", tmp_path / "b"], args=args, max_workers=1).run()
    merged = results[0]["result"]
    assert merged.n_events == 400
    sigma, error = merged.total()
    assert sigma == pytest.approx(10 * np.pi / 2, rel=1e-6)
    assert error > 0


def test_cross_sections_need_start_and_end(tmp_path):
    with write_run(tmp_path / "run", XS_QUANTITIES, output="Collisions",
                   config={"Modi": {"Collider": {"Impact": {"Max": 2.0}}}}) as w:
        initial = pp_initial(0.5)
        w.interaction(initial, initial, process=1)
        w.end(event=0)
    analysis = CrossSections(tmp_path / "run", analysis_args(CrossSections))
    with pytest.raises(ValueError, match="Print_Start_End"):
        analysis.run()


def test_cross_sections_empty_bins_are_interpolated(tmp_path):
    # every second bin inside the disc has no events; they must not count as P = 0
    bs = [(2 * k + 0.5) * 0.1 for k in range(5)]  # centres of bins 0, 2, 4, 6, 8 of width 0.1
    write_xs_run(tmp_path / "run", [(b, 1, (PROTON, PROTON)) for b in bs] + [(1.95, None, ())])
    analysis = CrossSections(tmp_path / "run", analysis_args(CrossSections, "--d-bins", "20"))
    analysis.run()
    sigma, _ = analysis.total()
    # P = 1 up to bin 8, interpolated to 0 at the next filled bin (19)
    centres = (np.arange(20) + 0.5) * 0.1
    prob = np.interp(centres, [0.05, 0.25, 0.45, 0.65, 0.85, 1.95], [1, 1, 1, 1, 1, 0])
    expected = 10 * np.pi * (prob * np.diff(np.linspace(0, 2, 21) ** 2)).sum()
    assert sigma == pytest.approx(expected)


# --- resonance reconstruction ---------------------------------------------

from brass.analyses.cross_sections import reconstructed_final_states  # noqa: E402

PI_P, PI_M, RHO0 = 211, -211, 113


def test_reconstruction_of_a_clean_decay():
    # p p -> n Delta++, Delta++ -> p pi+
    pdg = {0: PROTON, 1: PROTON, 2: NEUTRON, 3: DELTA_PP, 4: PROTON, 5: PI_P}
    interactions = [([0, 1], [2, 3]), ([3], [4, 5])]
    assert reconstructed_final_states(interactions, [2, 4, 5], pdg, {DELTA_PP}) == [(NEUTRON, DELTA_PP)]
    # not asked for
    assert reconstructed_final_states(interactions, [2, 4, 5], pdg, {RHO0}) == []


def test_reconstruction_through_a_chain_and_elastic_ids():
    # p p (elastic, IDs kept) -> p p -> p p rho0 ... rho0 -> pi+ pi-
    pdg = {0: PROTON, 1: PROTON, 2: PROTON, 3: PROTON, 4: RHO0, 5: PI_P, 6: PI_M}
    interactions = [([0, 1], [0, 1]), ([0, 1], [2, 3, 4]), ([4], [5, 6])]
    assert reconstructed_final_states(interactions, [2, 3, 5, 6], pdg, {RHO0}) == [(RHO0, PROTON, PROTON)]


def test_reconstruction_skips_rescattered_and_undecayed_resonances():
    pdg = {0: PROTON, 1: PROTON, 2: NEUTRON, 3: DELTA_PP, 4: PROTON, 5: PI_P}
    # the pi+ scatters elastically on the neutron afterwards: the Delta is not seen
    interactions = [([0, 1], [2, 3]), ([3], [4, 5]), ([5, 2], [5, 2])]
    assert reconstructed_final_states(interactions, [2, 4, 5], pdg, {DELTA_PP}) == []
    # the pi+ is absorbed into a new resonance with the neutron
    pdg[6] = 2214
    interactions = [([0, 1], [2, 3]), ([3], [4, 5]), ([5, 2], [6])]
    assert reconstructed_final_states(interactions, [4, 6], pdg, {DELTA_PP}) == []
    # a Delta that has not decayed is already in the final state
    assert reconstructed_final_states([([0, 1], [2, 3])], [2, 3], pdg, {DELTA_PP}) == []


def test_cross_sections_reconstructed_states(tmp_path):
    config = {"Modi": {"Collider": {"Sqrtsnn": 3.0, "Impact": {"Max": 2.0}}}}
    bs = (np.arange(100) + 0.5) * 1.0 / 100  # all inside 1 fm, all interact: sigma = 10 pi mb
    with write_run(tmp_path / "run", XS_QUANTITIES, output="Collisions", config=config) as w:
        for n, b in enumerate(bs):
            initial = pp_initial(b)
            w.particles(initial, event=n)
            nd = xs_particles((1, 0, 0, 0, NEUTRON, 0.1, 0, 0.5, M_P, 2), (1, 0, 0, 0, DELTA_PP, -0.1, 0, -0.5, 1.232, 3))
            w.interaction(initial, nd, process=3, sigma=31.4)
            delta = xs_particles((1, 0, 0, 0, DELTA_PP, -0.1, 0, -0.5, 1.232, 3))
            ppi = xs_particles((2, 0, 0, 0, PROTON, -0.1, 0, -0.3, M_P, 4), (2, 0, 0, 0, PI_P, 0, 0, -0.2, 0.138, 5))
            w.interaction(delta, ppi, process=5, sigma=0.0)
            final = xs_particles((3, 0, 0, 0, NEUTRON, 0.1, 0, 0.5, M_P, 2),
                                 (3, 0, 0, 0, PROTON, -0.1, 0, -0.3, M_P, 4), (3, 0, 0, 0, PI_P, 0, 0, -0.2, 0.138, 5))
            w.particles(final, event=n)
            w.end(event=n)
    analysis = CrossSections(tmp_path / "run", analysis_args(CrossSections, "--d-bins", "10"))
    analysis.run()
    sigma, _ = analysis.total()
    assert analysis.final_states()[tuple(sorted((NEUTRON, PROTON, PI_P)))][0] == pytest.approx(sigma)
    assert analysis.reconstructed_states() == {(NEUTRON, DELTA_PP): pytest.approx((sigma, sigma * 0.1))}
    assert CrossSections(tmp_path / "run", analysis_args(CrossSections, "--resonances")).resonances == frozenset()
