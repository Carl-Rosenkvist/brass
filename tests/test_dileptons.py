import numpy as np
import pytest
from brass import AnalysisRunner
from brass.analyses import Dileptons
from brass.testing import analysis_args, write_run

QUANTITIES = ["p0", "px", "py", "pz", "pdg"]

RHO_PDG = 113
CHANNEL = "113 -> -11 11"


def dilepton_args(*argv):
    return analysis_args(Dileptons, *argv)


def make_run(folder, n_events, weight, pz=0.4):
    """Each event: rho0 at rest -> e+ e- back to back, mass 2 * pz."""
    with write_run(
        folder, QUANTITIES, output="Dileptons", config={"Meta": {"test": 1}}
    ) as w:
        for event in range(n_events):
            w.interaction(
                incoming=[[2 * pz, 0.0, 0.0, 0.0, RHO_PDG]],
                outgoing=[
                    [pz, 0.0, 0.0, pz, -11],
                    [pz, 0.0, 0.0, -pz, 11],
                ],
                rho=0.0,
                sigma=weight,
                sigma_p=weight,
                process=5,
            )
            w.end(event=event)


def test_single_run_counts_events(tmp_path):
    run = tmp_path / "run"
    make_run(run, n_events=3, weight=0.5)

    analysis = Dileptons(run, dilepton_args())
    analysis.run()

    assert analysis.nevents == 3
    assert list(analysis.values) == [CHANNEL]
    assert analysis.values[CHANNEL].sum() == pytest.approx(3 * 0.5)

    # All pairs have invariant mass 0.8 and land in the same bin.
    edges = analysis.edges[CHANNEL]
    expected_bin = np.searchsorted(edges, 0.8, side="right") - 1
    assert np.flatnonzero(analysis.values[CHANNEL]).tolist() == [expected_bin]


def test_runs_with_same_meta_are_merged(tmp_path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    make_run(run1, n_events=3, weight=0.5)
    make_run(run2, n_events=2, weight=2.0)

    runner = AnalysisRunner(
        Dileptons,
        [run1, run2],
        args=dilepton_args(),
        max_workers=1,
    )

    results = runner.run()

    assert len(results) == 1

    merged = results[0]["result"]

    assert merged.nevents == 5
    assert merged.values[CHANNEL].sum() == pytest.approx(3 * 0.5 + 2 * 2.0)
