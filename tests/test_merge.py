from pathlib import Path
import random
import math
import yaml
import brass as br
from writing_utils import writeHeader, writeParticleBlock, writeEndBlock

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

def test_merge_events(tmp_path):
    # make two files with the SAME merge key -> they should combine
    bin_paths = []
    for i in range(2):
        run_dir = tmp_path / f"out-{i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        bin_path = run_dir / "particles_binary.bin"
        with open(bin_path, "wb") as f:
            events = [generate_particles(20), generate_particles(50)]
            write_binary(events, f)
        bin_paths.append(bin_path)

    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    br.run_analysis(
        file_and_meta=[(str(p), "Bogus=20") for p in bin_paths],  # same key -> merge
        analysis_names=["bulk"],
        quantities=["pdg", "p0", "px", "py", "pz"],
        output_folder=str(out_dir),
    )

    # --- read YAML output
    bulk_yaml = out_dir / "bulk.yaml"
    assert bulk_yaml.is_file(), "results/bulk.yaml not found"

    with open(bulk_yaml, "r") as f:
        data = yaml.safe_load(f)  # single doc written by this test

    assert data["n_events"] == 4  # 2 files * 2 events each
    spectra = data["spectra"][211]    # keys are strings in YAML
    merge_key = data["merge_key"]



def test_should_not_merge_events(tmp_path):
    # two separate binary files with identical event counts but DIFFERENT merge keys
    bin_paths = []
    for i in range(2):
        run_dir = tmp_path / f"out-{i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        bin_path = run_dir / "particles_binary.bin"
        with open(bin_path, "wb") as f:
            events = [generate_particles(20), generate_particles(50)]
            write_binary(events, f)
        bin_paths.append(bin_path)

    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Different merge keys -> should NOT merge
    file_and_meta = [(str(p), f"Bogus={idx}") for idx, p in enumerate(bin_paths)]

    br.run_analysis(
        file_and_meta=file_and_meta,
        analysis_names=["bulk"],
        quantities=["pdg", "p0", "px", "py", "pz"],
        output_folder=str(out_dir),
    )

    bulk_yaml = out_dir / "bulk.yaml"
    assert bulk_yaml.is_file(), "results/bulk.yaml not found"

    # Two separate YAML documents expected
    with open(bulk_yaml, "r") as f:
        docs = list(yaml.safe_load_all(f))

    assert len(docs) == 2, f"expected 2 docs (no merge), got {len(docs)}"

    # Each doc corresponds to one file -> 2 events each
    for doc in docs:
        assert doc["n_events"] == 2

        # spectra keys are strings
        _ = doc["spectra"][211]  # access to ensure it exists

    # Merge keys must differ (and, if map-like, Bogus should be 0 and 1)
    mk0, mk1 = docs[0]["merge_key"], docs[1]["merge_key"]
    assert mk0 != mk1, f"merge keys unexpectedly equal: {mk0}"

    # If your to_yaml emits a mapping, check Bogus specifically:
    if isinstance(mk0, dict) and isinstance(mk1, dict):
        assert mk0.get("Bogus") in (0, "0")
        assert mk1.get("Bogus") in (1, "1")
