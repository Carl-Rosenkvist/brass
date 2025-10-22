import random
import math
import yaml
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
    # --- setup run directory and files
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

    br.run_analysis(
        file_and_meta=[(str(bin_path), "Bogus=20")],
        analysis_names=["bulk"],
        quantities=["pdg", "p0", "px", "py", "pz"],
        output_folder=str(out_dir),
    )

    # --- read YAML output
    bulk_yaml = out_dir / "bulk.yaml"
    assert bulk_yaml.is_file(), "results/bulk.yaml not found"

    with open(bulk_yaml, "r") as f:
        data = yaml.safe_load(f)

    spectra = data["spectra"][211]
    n_events = data["n_events"]
    merge_key = data["merge_key"]

    # --- validate metadata
    assert merge_key["Bogus"] == 20
    assert n_events == len(events)
    assert "values" in spectra

    # --- reshape histogram
    counts = np.array(spectra["values"]).reshape(spectra["pt_bins"], spectra["y_bins"])
    pt_edges = np.linspace(*spectra["pt_range"], spectra["pt_bins"] + 1)
    y_edges = np.linspace(*spectra["y_range"], spectra["y_bins"] + 1)

    # --- projections from analysis
    dn_dy_analysis = counts.sum(axis=0) * (pt_edges[1] - pt_edges[0])
    dn_dpt_analysis = counts.sum(axis=1) * (y_edges[1] - y_edges[0])

    # --- compute from truth (the generated particles)
    all_parts = [p for ev in events for p in ev]
    px = np.array([p[2] for p in all_parts])
    py = np.array([p[3] for p in all_parts])
    pz = np.array([p[4] for p in all_parts])
    e = np.array([p[1] for p in all_parts])
    pt = np.sqrt(px**2 + py**2)
    y = 0.5 * np.log((e + pz) / (e - pz))

    dn_dy_truth, _ = np.histogram(y, bins=y_edges)
    dn_dpt_truth, _ = np.histogram(pt, bins=pt_edges)
    dn_dy_truth = dn_dy_truth / (n_events * (y_edges[1] - y_edges[0]))
    dn_dpt_truth = dn_dpt_truth / (n_events * (pt_edges[1] - pt_edges[0]))

    # --- consistency checks (integrated counts and shape correlation)
    assert np.isclose(dn_dy_analysis.sum(), dn_dy_truth.sum(), rtol=10e-5)
    assert np.isclose(dn_dpt_analysis.sum(), dn_dpt_truth.sum(), rtol=10e-5)

    # normalized correlation to test shape similarity
    corr_y = np.corrcoef(dn_dy_analysis, dn_dy_truth)[0, 1]
    corr_pt = np.corrcoef(dn_dpt_analysis, dn_dpt_truth)[0, 1]

    # --- basic shape sanity
    assert corr_y > 0.999, f"Low correlation in dN/dy: {corr_y}"
    assert corr_pt > 0.999, f"Low correlation in dN/dpT: {corr_pt}"

    # --- optional logging for debug output
    print(f"Total counts (analysis): {counts.sum()}")
    print(f"Correlation dN/dy = {corr_y:.3f}, dN/dpT = {corr_pt:.3f}")
