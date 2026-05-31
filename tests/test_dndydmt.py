import random
import math
import numpy as np
import brass as br

from writing_utils import (
    writeHeader,
    writeParticleBlock,
    writeEndBlock,
)


def generate_particles(n, mass=0.139):
    """Generate simple random pion-like particles."""
    parts = []
    for _ in range(n):
        px = random.uniform(-0.5, 0.5)
        py = random.uniform(-0.5, 0.5)
        pz = random.uniform(-0.5, 0.5)
        p0 = math.sqrt(mass * mass + px * px + py * py + pz * pz)
        pdg = 211
        parts.append((pdg, p0, px, py, pz))
    return parts


def write_binary(events, f):
    writeHeader(f)
    impact_parameter = 0.0
    ensemble_number = 0

    for i, event in enumerate(events):
        writeParticleBlock(f, i, ensemble_number, event)
        writeEndBlock(f, i, ensemble_number, impact_parameter, empty=False)


def test_dndydmt(tmp_path):
    random.seed(12345)

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    bin_path = run_dir / "particles_binary.bin"
    with open(bin_path, "wb") as f:
        events = [
            generate_particles(30),
            generate_particles(40),
        ]
        write_binary(events, f)

    meta_label = "Test=1"
    state = br.run_analysis_one_file(
        filename=str(bin_path),
        meta=meta_label,
        analysis_name="dndydmt",
        quantities=["pdg", "p0", "px", "py", "pz"],
        opts=None,
    )

    d = state[meta_label]["dndydmt"]

    n_events = d["n_events"]
    assert n_events == len(events)

    H = d["per_pdg"][211]

    # --- Project using HistND ---
    counts_y, (y_edges,) = H.project([1])
    counts_mt, (mt_edges,) = H.project([0])

    y_edges = np.asarray(y_edges)
    mt_edges = np.asarray(mt_edges)

    dy = np.diff(y_edges)
    dmt = np.diff(mt_edges)

    dn_dy_analysis = counts_y / (n_events * dy)
    dn_dmt_analysis = counts_mt / (n_events * dmt)

    # --- Compute truth distributions ---
    parts = [p for ev in events for p in ev]

    px = np.array([p[2] for p in parts])
    py = np.array([p[3] for p in parts])
    pz = np.array([p[4] for p in parts])
    e = np.array([p[1] for p in parts])

    pt = np.sqrt(px**2 + py**2)
    m2 = np.maximum(e**2 - (px**2 + py**2 + pz**2), 0.0)
    m = np.sqrt(m2)

    mt = np.hypot(pt, m)
    y = 0.5 * np.log((e + pz) / (e - pz))

    truth_mt, _ = np.histogram(mt, bins=mt_edges)
    truth_y, _ = np.histogram(y, bins=y_edges)

    dn_dmt_truth = truth_mt / (n_events * dmt)
    dn_dy_truth = truth_y / (n_events * dy)

    np.testing.assert_allclose(dn_dy_analysis, dn_dy_truth)
    np.testing.assert_allclose(dn_dmt_analysis, dn_dmt_truth)
