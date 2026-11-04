import argparse
import struct

import numpy as np
import pytest
from brass import AnalysisRunner
from brass.analyses import Dileptons

QUANTITIES = ["p0", "px", "py", "pz", "pdg"]

RHO_PDG = 113
CHANNEL = "113 -> -11 11"


def dilepton_args(*argv):
    """Parse Dileptons arguments with the real CLI defaults."""
    parser = argparse.ArgumentParser()
    Dileptons.add_arguments(parser)
    return parser.parse_args(list(argv))


def write_header(file):
    version = b"SMASH-3.1"
    file.write(b"SMSH")
    file.write(struct.pack("<HHI", 9, 1, len(version)))
    file.write(version)


def write_particle(file, p0, px, py, pz, pdg):
    file.write(struct.pack("<ddddi", p0, px, py, pz, pdg))


def write_dilepton_decay(file, pz, weight):
    """rho0 at rest -> e+ e- back to back, invariant mass 2 * pz."""
    file.write(b"i")
    file.write(struct.pack("<ii", 1, 2))
    file.write(struct.pack("<dddi", 0.0, weight, weight, 5))

    write_particle(file, 2 * pz, 0.0, 0.0, 0.0, RHO_PDG)
    write_particle(file, pz, 0.0, 0.0, pz, -11)
    write_particle(file, pz, 0.0, 0.0, -pz, 11)


def write_end(file, event_number):
    file.write(b"f")
    file.write(struct.pack("<Iid?", event_number, 0, 0.0, False))


def make_run(folder, n_events, weight):
    folder.mkdir()

    quantities = "\n".join(f"      - {q}" for q in QUANTITIES)
    (folder / "config.yaml").write_text(
        f"""
Meta:
  test: 1

Output:
  Dileptons:
    Quantities:
{quantities}
"""
    )

    with (folder / "Dileptons_custom.bin").open("wb") as file:
        write_header(file)

        for event in range(n_events):
            write_dilepton_decay(file, pz=0.4, weight=weight)
            write_end(file, event)


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
