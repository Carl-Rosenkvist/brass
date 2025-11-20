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
        p0 = math.sqrt(mass*mass + px*px + py*py + pz*pz)
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
    # --- setup directories ---
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    bin_path = run_dir / "particles_binary.bin"
    with open(bin_path, "wb") as f:
        events = [
            generate_particles(30),
            generate_particles(40),
        ]
        write_binary(events, f)

    # --- run analysis ---
    meta_label = "Test=1"
    state = br.run_analysis_one_file(
        filename=str(bin_path),
        meta=meta_label,
        analysis_name="dndydmt",
        quantities=["pdg", "p0", "px", "py", "pz"],
        opts=None,
    )

    assert meta_label in state
    assert "dndydmt" in state[meta_label]
    d = state[meta_label]["dndydmt"]

    n_events = d["n_events"]
    assert n_events == len(events)

    H = d["incl"]
    counts = np.array(H.counts)
    mt_edges = np.array(H.edges[0])
    y_edges = np.array(H.edges[1])

    # --- Compute bin widths ---
    dmt = mt_edges[1] - mt_edges[0]
    dy = y_edges[1] - y_edges[0]

    # --- Compute inclusive distributions ---
    N_y = counts.sum(axis=0)      # integrate over mt
    N_mt = counts.sum(axis=1)     # integrate over y

    dn_dy_analysis = N_y / (n_events * dy * dmt)
    dn_dmt_analysis = N_mt / (n_events * dy * dmt)

    # --- Compute truth distributions ---
    parts = [p for ev in events for p in ev]
    px = np.array([p[2] for p in parts])
    py = np.array([p[3] for p in parts])
    pz = np.array([p[4] for p in parts])
    e  = np.array([p[1] for p in parts])

    pt = np.sqrt(px**2 + py**2)
    m2 = np.maximum(e*e - (px*px + py*py + pz*pz), 0.0)
    m = np.sqrt(m2)
    mt = np.hypot(pt, m)
    y = 0.5 * np.log((e + pz) / (e - pz))

    truth_mt, _ = np.histogram(mt, bins=mt_edges)
    truth_y,  _ = np.histogram(y,  bins=y_edges)

    dn_dmt_truth = truth_mt / (n_events * dmt * dy)
    dn_dy_truth  = truth_y  / (n_events * dmt * dy)

    # --- Compare totals ---
    assert np.isclose(dn_dy_analysis.sum(),  dn_dy_truth.sum(),  rtol=1e-4)
    assert np.isclose(dn_dmt_analysis.sum(), dn_dmt_truth.sum(), rtol=1e-4)

    # --- Compare shapes via correlation ---
    corr_y  = np.corrcoef(dn_dy_analysis,  dn_dy_truth)[0, 1]
    corr_mt = np.corrcoef(dn_dmt_analysis, dn_dmt_truth)[0, 1]

    assert corr_y > 0.999,  f"Low correlation in dN/dy:   {corr_y}"
    assert corr_mt > 0.999, f"Low correlation in dN/dmT: {corr_mt}"

    print("dndydmt test OK")
    print(f"Correlation dN/dy  = {corr_y:.3f}")
    print(f"Correlation dN/dmT = {corr_mt:.3f}")
