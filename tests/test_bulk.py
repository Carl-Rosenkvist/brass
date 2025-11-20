import random
import math
import numpy as np
import brass as br

from writing_utils import (
    writeHeader,
    writeParticleBlock,
    writeEndBlock,
    write_config_yaml,
)


def make_config_yaml_for_bulk():
    return {
        "Modi.Collider.Sqrtsnn": 17.3,
        "Modi.Collider.Projectile.Particles.2212": 82,
        "Modi.Collider.Projectile.Particles.2112": 126,
        "Modi.Collider.Target.Particles.2212": 82,
        "Modi.Collider.Target.Particles.2112": 126,
        "Output.Particles.Quantities": ["pdg", "p0", "px", "py", "pz"],
    }


def generate_particles(n, mass=0.139):
    parts = []
    for _ in range(n):
        px = random.uniform(-0.5, 0.5)
        py = random.uniform(-0.5, 0.5)
        pz = random.uniform(-0.5, 0.5)
        p0 = math.sqrt(mass * mass + px * px + py * py + pz * pz)
        pdg = 211
        parts.append((pdg, p0, px, py, pz))
    return parts


def write_binary(events, bfile):
    writeHeader(bfile)
    impact_parameter = 0.0
    ensemble_number = 0
    for i, event in enumerate(events):
        writeParticleBlock(bfile, i, ensemble_number, event)
        writeEndBlock(bfile, i, ensemble_number, impact_parameter, empty=False)


def test_bulk_distributions(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg_path = run_dir / "config.yaml"
    write_config_yaml(str(cfg_path), make_config_yaml_for_bulk())

    bin_path = run_dir / "particles_binary.bin"
    with open(bin_path, "wb") as f:
        events = [generate_particles(20), generate_particles(50)]
        write_binary(events, f)

    out_dir = tmp_path / "results"
    out_dir.mkdir()

    meta_label = "Bogus=20"
    state = br.run_analysis_one_file(
        filename=str(bin_path),
        meta=meta_label,
        analysis_name="bulk",
        quantities=["pdg", "p0", "px", "py", "pz"],
        opts=None,
    )

    assert meta_label in state
    assert "bulk" in state[meta_label]
    bulk_state = state[meta_label]["bulk"]

    n_events = bulk_state["n_events"]
    assert n_events == len(events)

    spectra = bulk_state["spectra"][211]

    x_edges = np.array(spectra["x_edges"])
    y_edges = np.array(spectra["y_edges"])
    x_bins = spectra["x_bins"]
    y_bins = spectra["y_bins"]
    counts = np.array(spectra["counts"]).reshape(x_bins, y_bins)

    pt_edges = x_edges
    y_edges_ = y_edges

    dpt = pt_edges[1] - pt_edges[0]
    dy = y_edges_[1] - y_edges_[0]

    N_y = counts.sum(axis=0)
    N_pt = counts.sum(axis=1)

    dn_dy_analysis = N_y / (n_events * dy)
    dn_dpt_analysis = N_pt / (n_events * dpt)

    all_parts = [p for ev in events for p in ev]
    px = np.array([p[2] for p in all_parts])
    py = np.array([p[3] for p in all_parts])
    pz = np.array([p[4] for p in all_parts])
    e = np.array([p[1] for p in all_parts])
    pt = np.sqrt(px**2 + py**2)
    y = 0.5 * np.log((e + pz) / (e - pz))

    dn_dy_truth, _ = np.histogram(y, bins=y_edges_)
    dn_dpt_truth, _ = np.histogram(pt, bins=pt_edges)
    dn_dy_truth = dn_dy_truth / (n_events * dy)
    dn_dpt_truth = dn_dpt_truth / (n_events * dpt)

    assert np.isclose(dn_dy_analysis.sum(), dn_dy_truth.sum(), rtol=1e-4)
    assert np.isclose(dn_dpt_analysis.sum(), dn_dpt_truth.sum(), rtol=1e-4)

    corr_y = np.corrcoef(dn_dy_analysis, dn_dy_truth)[0, 1]
    corr_pt = np.corrcoef(dn_dpt_analysis, dn_dpt_truth)[0, 1]

    assert corr_y > 0.999, f"Low correlation in dN/dy: {corr_y}"
    assert corr_pt > 0.999, f"Low correlation in dN/dpT: {corr_pt}"

    print(f"Total counts (analysis): {counts.sum()}")
    print(f"Correlation dN/dy = {corr_y:.3f}, dN/dpT = {corr_pt:.3f}")
