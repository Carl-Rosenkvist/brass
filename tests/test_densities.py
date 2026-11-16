"""Thermodynamics output at a point and the position grid."""

import numpy as np
import pytest
import yaml

from brass import AnalysisRunner
from brass.analyses import PositionGrid, Thermodynamics
from brass.testing import analysis_args, write_run
from brass.thermodynamics import read_thermodynamics

THERMO = """\
# SMASH-3.4 thermodynamics output
# @ point (  0.00,   0.00,   0.00) [fm]
# net_baryon
# time [fm], rho_eckart [fm^-3], tmn [GeV/fm^3],
# event 0
 -1.00  0.100000000000  1.0
  0.00  0.300000000000  2.0
  1.00  0.200000000000  3.0
# event 1
 -1.00  0.300000000000  1.0
  0.00  0.500000000000  2.0
  1.00  0.200000000000  5.0
"""


def thermo_run(folder):
    folder.mkdir(parents=True)
    (folder / "thermodynamics.dat").write_text(THERMO)
    (folder / "config.yaml").write_text(yaml.safe_dump({"General": {"Modus": "Collider"}}))
    return folder


def test_read_thermodynamics(tmp_path):
    data = read_thermodynamics(thermo_run(tmp_path / "run") / "thermodynamics.dat")
    assert data.smash_version == "SMASH-3.4"
    assert data.position == (0.0, 0.0, 0.0)
    assert data.density_type == "net_baryon"
    assert data.quantities == ["rho_eckart", "tmn"]
    assert len(data.events) == 2
    np.testing.assert_allclose(data.events[1].values[:, 0], [0.3, 0.5, 0.2])


def test_thermodynamics_analysis(tmp_path):
    a = Thermodynamics(thermo_run(tmp_path / "a"), analysis_args(Thermodynamics))
    a.run()
    assert a.events == 2
    np.testing.assert_allclose(a.times, [-1, 0, 1])
    np.testing.assert_allclose(a.mean("rho_eckart"), [0.2, 0.4, 0.2])
    # values 0.1 and 0.3: sample std 0.1414, error 0.1
    np.testing.assert_allclose(a.error("rho_eckart"), [0.1, 0.1, 0.0], atol=1e-12)
    np.testing.assert_allclose(a.mean("tmn"), [1, 2, 4])

    thermo_run(tmp_path / "b")
    results = AnalysisRunner(Thermodynamics, [tmp_path / "a", tmp_path / "b"],
                             args=analysis_args(Thermodynamics), max_workers=1).run()
    merged = results[0]["result"]
    assert merged.events == 4
    np.testing.assert_allclose(merged.mean("rho_eckart"), [0.2, 0.4, 0.2])


def test_not_a_thermodynamics_file(tmp_path):
    (tmp_path / "x.dat").write_text("1 2 3\n")
    with pytest.raises(ValueError, match="thermodynamics"):
        read_thermodynamics(tmp_path / "x.dat")


QUANTITIES = ["t", "x", "y", "pdg"]


def positions(t, xy, pdg=211):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return {"t": np.full(len(xy), t), "x": xy[:, 0], "y": xy[:, 1], "pdg": np.full(len(xy), pdg, np.int32)}


def test_position_grid(tmp_path):
    with write_run(tmp_path / "run", QUANTITIES, config={"Modi": {"Box": {"Length": 4.0}}}) as w:
        for event in range(2):
            w.particles(positions(0.0, [(0.5, 0.5), (0.6, 0.4), (3.5, 1.5)]), event=event)
            w.particles(positions(50.0, [(1.5, 2.5)]), event=event)
            w.end(event=event)
    a = PositionGrid(tmp_path / "run", analysis_args(PositionGrid, "--bins", "4"))
    a.run()
    assert a.events == 2
    np.testing.assert_allclose(a.times, [0, 50])
    cell_volume = 1.0 * 1.0 * 4.0
    first = a.density(0)
    assert first[0, 0] == pytest.approx(2 / cell_volume)   # two particles per event in cell (0, 0)
    assert first[3, 1] == pytest.approx(1 / cell_volume)
    assert first.sum() * cell_volume == pytest.approx(3)
    assert a.density(1)[1, 2] == pytest.approx(1 / cell_volume)


def test_position_grid_pdgs_and_length(tmp_path):
    with write_run(tmp_path / "run", QUANTITIES) as w:
        both = {k: np.concatenate([positions(0.0, [(0.5, 0.5)], 211)[k], positions(0.0, [(0.5, 0.5)], 2212)[k]])
                for k in QUANTITIES}
        w.particles(both, event=0)
        w.end(event=0)
    with pytest.raises(ValueError, match="--length"):
        PositionGrid(tmp_path / "run", analysis_args(PositionGrid))
    a = PositionGrid(tmp_path / "run", analysis_args(PositionGrid, "--length", "2", "--bins", "2", "--pdgs", "2212"))
    a.run()
    assert a.grids[0][0, 0] == 1
