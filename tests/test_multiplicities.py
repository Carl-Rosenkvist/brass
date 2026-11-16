import numpy as np
import pytest

from brass import AnalysisRunner
from brass.analyses import Multiplicities
from brass.testing import analysis_args, write_run

QUANTITIES = ["p0", "px", "py", "pz", "pdg"]
PIP, PIM = 211, -211


def event(n_plus, n_minus, pt_plus=0.3):
    rows = [(1.0, pt_plus, 0.0, 0.1, PIP)] * n_plus + [(1.0, 0.0, 0.2, 0.1, PIM)] * n_minus + [(1.0, 0, 0, 0, 2212)]
    data = np.array(rows, dtype=float)
    return {q: data[:, i].astype(np.int32 if q == "pdg" else float) for i, q in enumerate(QUANTITIES)}


def write(folder, events, testparticles=1):
    with write_run(folder, QUANTITIES, config={"General": {"Testparticles": testparticles}}) as w:
        for n, e in enumerate(events):
            w.particles(e, event=n)
            w.end(event=n)


def test_multiplicities_and_errors(tmp_path):
    # pi+ counts 2, 4, 6 -> mean 4, sample std 2, error 2 / sqrt(3)
    write(tmp_path / "run", [event(2, 1, 0.2), event(4, 0, 0.4), event(6, 1, 0.6)])
    a = Multiplicities(tmp_path / "run", analysis_args(Multiplicities, "--pdgs", "211", "-211"))
    a.run()
    assert a.events == 3
    mean, error = a.multiplicity(PIP)
    assert mean == pytest.approx(4.0) and error == pytest.approx(2 / np.sqrt(3))
    assert a.multiplicity(PIM)[0] == pytest.approx(2 / 3)
    # mean pT: average of the event means 0.2, 0.4, 0.6
    pt, pt_error = a.mean_pt(PIP)
    assert pt == pytest.approx(0.4) and pt_error == pytest.approx(0.2 / np.sqrt(3))
    # pi-: only two events have one; both with pT 0.2
    assert a.mean_pt(PIM) == (pytest.approx(0.2), pytest.approx(0.0))


def test_test_particles_and_merge(tmp_path):
    write(tmp_path / "a", [event(20, 0), event(40, 0)], testparticles=10)
    write(tmp_path / "b", [event(60, 0)], testparticles=10)
    args = analysis_args(Multiplicities, "--pdgs", "211")
    results = AnalysisRunner(Multiplicities, [tmp_path / "a", tmp_path / "b"], args=args, max_workers=1).run()
    merged = results[0]["result"]
    assert merged.events == 3 and merged.testparticles == 10
    mean, error = merged.multiplicity(PIP)
    assert mean == pytest.approx(4.0) and error == pytest.approx(2 / np.sqrt(3))
