import struct

import numpy as np
import pytest

from brass import AnalysisRunner
from brass.analyses import DileptonSpectra
from brass.testing import analysis_args, write_run

QUANTITIES = ["p0", "px", "py", "pz", "pdg", "pdg_mother1"]
M_E = 0.000511


def particle(pdg, p3, mass, mother=0):
    p3 = np.asarray(p3, float)
    return (np.sqrt(mass**2 + p3 @ p3), *p3, pdg, mother)


def rows(*particles):
    data = np.array(particles, dtype=float)
    return {q: data[:, i].astype(np.int32 if q in ("pdg", "pdg_mother1") else float)
            for i, q in enumerate(QUANTITIES)}


def two_body(parent, mother, mass, pz, weight):
    """parent at rest in x-y, moving along z, decaying into e+ e- along x."""
    e_cm = mass / 2
    k = np.sqrt(e_cm**2 - M_E**2)
    p_parent = rows(particle(parent, (0, 0, pz), mass, mother))
    # boost of the leptons along z: give both the same pz share
    leptons = rows(particle(11, (k, 0.0, pz / 2), M_E), particle(-11, (-k, 0.0, pz / 2), M_E))
    return p_parent, leptons, weight


def write_events(folder, events):
    with write_run(folder, QUANTITIES, output="Dileptons") as w:
        for n, decays in enumerate(events):
            for parent, leptons, weight in decays:
                w.interaction(parent, leptons, sigma=weight, process=5)
            w.end(event=n)


def test_spectra_per_channel_and_mother(tmp_path):
    rho_from_pions = two_body(113, 211, 0.70, 0.5, 2e-5)
    rho_from_omega = two_body(113, 223, 0.60, 0.4, 1e-5)
    omega = two_body(223, 0, 0.782, 0.6, 4e-5)
    write_events(tmp_path / "run", [[rho_from_pions, omega], [rho_from_omega]])

    a = DileptonSpectra(tmp_path / "run", analysis_args(DileptonSpectra))
    a.run()
    assert a.events == 2
    assert a.channels() == ["113 -> -11 11", "223 -> -11 11"]
    assert a.mothers("113 -> -11 11") == [211, 223]

    dm = 2.0 / 150
    centres, rho = a.spectrum("mass", channels=["113 -> -11 11"])
    assert rho.sum() * dm == pytest.approx((2e-5 + 1e-5) / 2)
    assert centres[np.argmax(rho)] == pytest.approx(0.70, abs=dm)
    _, rho_no_omega = a.spectrum("mass", channels=["113 -> -11 11"], exclude_mothers=(223,))
    assert rho_no_omega.sum() * dm == pytest.approx(2e-5 / 2)
    _, total = a.spectrum("mass")
    assert total.sum() * dm == pytest.approx(7e-5 / 2)

    # pair momentum split and mass windows
    _, above = a.spectrum("mass_above")
    _, below = a.spectrum("mass_below")
    assert above.sum() * dm == pytest.approx(0.0)  # all pairs have |p| <= 0.6
    assert below.sum() * dm == pytest.approx(7e-5 / 2)
    dpt = 2.0 / 80
    _, window2 = a.spectrum("pt_2")  # 0.47 < M < 0.70: the rho from the omega only (0.70 is not inside)
    assert window2.sum() * dpt == pytest.approx(1e-5 / 2)
    assert a.window_label(3) == "M > 0.7 GeV"


def write_constant_acc(path, value):
    with open(path, "wb") as f:
        f.write(b"constant".ljust(80))
        f.write(struct.pack("<9f", 3, 3, 3, 17, 3, 1, 0.0035, 0.0035, 0.02))
        for pid in (3, 2):
            f.write(struct.pack("<i", pid))
            f.write(struct.pack("<3i", 10, 9, 6))
            f.write(struct.pack("<6f", 0.0, 2.0, 0.0, 90.0, 0.0, 60.0))
            f.write(np.full(10 * 9 * 6, value, "<f4").tobytes())
    return path


def hades_args(acc, *extra):
    return analysis_args(DileptonSpectra, "--acceptance", str(acc), "--seed", "1", *extra)


def forward_pair(mass, angle_deg, pz=0.6):
    """Leptons in the x-z plane at +-angle/2 around the beam (inside the acceptance)."""
    half = np.radians(angle_deg / 2)
    p = 0.3
    parent = rows(particle(113, (0, 0, 2 * p * np.cos(half)), mass))
    leptons = rows(particle(11, (p * np.sin(half), 0.0, p * np.cos(half)), M_E),
                   particle(-11, (-p * np.sin(half), 0.0, p * np.cos(half)), M_E))
    return parent, leptons, 1.0


def test_hades_filter_weights_and_cuts(tmp_path):
    acc = write_constant_acc(tmp_path / "half.acc", 0.5)
    write_events(tmp_path / "run", [[forward_pair(0.3, 60.0)], [forward_pair(0.3, 4.0)]])

    plain = DileptonSpectra(tmp_path / "run", analysis_args(DileptonSpectra))
    plain.run()
    filtered = DileptonSpectra(tmp_path / "run", hades_args(acc))
    filtered.run()
    dm = 2.0 / 150
    assert plain.spectrum()[1].sum() * dm == pytest.approx(2 / 2)
    # the 4 degree pair is cut by the opening angle; the other gets 0.5 * 0.5
    assert filtered.spectrum()[1].sum() * dm == pytest.approx(0.25 / 2, rel=1e-6)

    narrow = DileptonSpectra(tmp_path / "run", hades_args(acc, "--lepton-momentum", "0.5", "2.0"))
    narrow.run()
    assert narrow.spectrum()[1].sum() == 0.0


def test_merge(tmp_path):
    for name in ("a", "b"):
        write_events(tmp_path / name, [[two_body(113, 0, 0.7, 0.5, 1e-5)]])
    args = analysis_args(DileptonSpectra)
    results = AnalysisRunner(DileptonSpectra, [tmp_path / "a", tmp_path / "b"], args=args, max_workers=1).run()
    merged = results[0]["result"]
    assert merged.events == 2
    assert merged.spectrum()[1].sum() * 2.0 / 150 == pytest.approx(1e-5)
