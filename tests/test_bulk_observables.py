import math

import numpy as np
import pytest

import brass
from brass import AnalysisRunner
from brass.analyses import BulkObservables
from brass.testing import analysis_args, concatenate_particles, generate_particles, write_run

PION, PROTON, SIGMA0 = 211, 2212, 3212
QUANTITIES = ["p0", "px", "py", "pz", "pdg"]


def make_events(rng):
    """Three events with pions, protons and a few Σ⁰."""
    events = []
    for _ in range(3):
        events.append(
            concatenate_particles(
                generate_particles(rng, 3000, pdg=PION, mass=0.138, y_sigma=2.5, pt_mean=0.4),
                generate_particles(rng, 1000, pdg=PROTON, mass=0.938, y_mean=0.3, y_sigma=3.0, pt_mean=0.6),
                generate_particles(rng, 100, pdg=SIGMA0, mass=1.193, y_sigma=1.0, pt_mean=0.5),
            )
        )
    return events


def write(folder, events):
    with write_run(folder, QUANTITIES) as writer:
        for i, event in enumerate(events):
            writer.particles(event, event=i)


def truth(events, pdgs, midrapidity=0.5):
    all_ = concatenate_particles(*events)
    sel = np.isin(all_["pdg"], pdgs)
    y, pt, mt, m = all_["y"][sel], all_["pt"][sel], all_["mt"][sel], all_["mass"][sel]
    mid = np.abs(y) < midrapidity
    return {
        "total": sel.sum(),
        "mid": mid.sum(),
        "meanpt": pt[mid].mean(),
        "meanmt": (mt - m)[mid].mean(),
        "y": y,
        "pt_mid": pt[mid],
        "mt0_mid": (mt - m)[mid],
    }


def run(tmp_path, events, *argv):
    write(tmp_path / "run", events)
    args = analysis_args(BulkObservables, "--pdgs", str(PION), str(PROTON), *argv)
    analysis = BulkObservables(tmp_path / "run", args)
    analysis.run()
    return analysis


def test_observables_are_exact(tmp_path):
    events = make_events(np.random.default_rng(1))
    analysis = run(tmp_path, events)
    assert analysis.events == 3

    for pdg in (PION, PROTON):
        t = truth(events, [pdg])
        obs = analysis.observables(pdg)
        # protons have |y| > 4 too: the 4π multiplicity must still count them all
        assert obs["total_multiplicity"] == pytest.approx(t["total"] / 3)
        assert obs["midrapidity_yield"] == pytest.approx(t["mid"] / 3)
        assert obs["meanpt"] == pytest.approx(t["meanpt"], rel=1e-9)
        assert obs["meanmt"] == pytest.approx(t["meanmt"], rel=1e-6)

        y_edges = np.linspace(-4, 4, 41)
        x, dndy = obs["yspectra"]
        np.testing.assert_allclose(dndy, np.histogram(t["y"], y_edges)[0] / 3 / 0.2)

        edges = np.array(analysis.pt_edges)
        x, spectrum = obs["ptspectra"]
        np.testing.assert_allclose(spectrum, np.histogram(t["pt_mid"], edges)[0] / 3 / np.diff(edges))
        x, spectrum = obs["mtspectra"]
        np.testing.assert_allclose(
            spectrum, np.histogram(t["mt0_mid"], edges)[0] / 3 / np.diff(edges), atol=1e-12
        )


def test_add_pdgs_and_lists_of_pdgs(tmp_path):
    events = make_events(np.random.default_rng(2))
    write(tmp_path / "a", events)
    args = analysis_args(BulkObservables, "--pdgs", str(PROTON), str(PION), "--add-pdgs", str(PROTON), str(SIGMA0))
    analysis = BulkObservables(tmp_path / "a", args)
    analysis.run()
    t = truth(events, [PROTON, SIGMA0])
    assert analysis.observables(PROTON)["total_multiplicity"] == pytest.approx(t["total"] / 3)
    both = analysis.observables([PROTON, PION])
    assert both["total_multiplicity"] == pytest.approx(truth(events, [PROTON, SIGMA0, PION])["total"] / 3)


def test_merging_runs(tmp_path):
    rng = np.random.default_rng(3)
    first, second = make_events(rng), make_events(rng)
    write(tmp_path / "r1", first)
    write(tmp_path / "r2", second)
    args = analysis_args(BulkObservables, "--pdgs", str(PION))
    results = AnalysisRunner(BulkObservables, [tmp_path / "r1", tmp_path / "r2"], args=args, max_workers=1).run()
    assert len(results) == 1
    merged = results[0]["result"]
    assert merged.events == 6
    t = truth(first + second, [PION])
    assert merged.observables(PION)["meanpt"] == pytest.approx(t["meanpt"], rel=1e-9)
    assert merged.observables(PION)["total_multiplicity"] == pytest.approx(t["total"] / 6)


def test_no_particles_at_midrapidity_gives_nan(tmp_path):
    rng = np.random.default_rng(4)
    event = generate_particles(rng, 10, pdg=PION, mass=0.138, y_mean=3.0, y_sigma=0.01)
    analysis = run(tmp_path, [event])
    assert math.isnan(analysis.observables(PION)["meanpt"])
    assert analysis.observables(PROTON)["total_multiplicity"] == 0


def test_mt0_histogram(tmp_path):
    """The computed quantity mt0 = mT - m can be histogrammed and used as weight."""
    rng = np.random.default_rng(5)
    particles = generate_particles(rng, 500, pdg=PION, mass=0.138)
    write(tmp_path / "q", [particles])
    reader = brass.BinaryReader(str(tmp_path / "q" / "particles_binary.bin"), QUANTITIES)
    request = brass.HistogramRequest()
    request.quantities = ["mt0"]
    request.axes = [brass.RegularAxis(1, 0.0, 100.0)]
    request.weight_quantity = "mt0"
    result = brass.histogram(reader, request)
    assert result.values[0] == pytest.approx((particles["mt"] - particles["mass"]).sum(), rel=1e-6)


def test_bad_options(tmp_path):
    write(tmp_path / "b", make_events(np.random.default_rng(6))[:1])
    with pytest.raises(ValueError, match="--add-pdgs"):
        BulkObservables(tmp_path / "b", analysis_args(BulkObservables, "--pdgs", "211", "--add-pdgs", "3122", "3212"))
    with pytest.raises(ValueError, match="increasing"):
        BulkObservables(tmp_path / "b", analysis_args(BulkObservables, "--pdgs", "211", "--pt-edges", "1", "0.5"))
