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

