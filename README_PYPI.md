# BRASS — Binary Reader & Analysis Suite (Python)

Lightweight Python tools to read and analyze binary particle output files. Designed primarily for SMASH BINARY outputs.
## Features
- Pure-Python API for reading particle binary files
- Simple analysis runner (from Python or CLI)
- Minimal dependencies (pybind11 optional for native extensions)
- Developed primarily for SMASH, but usable elsewhere

## Quick Start
```python
import numpy as np
from brass import BinaryReader, CollectorAccessor

accessor = CollectorAccessor()
reader = BinaryReader("particles_binary.bin", ["pdg", "pz", "p0"], accessor)
reader.read()

pz  = accessor.get_double_array("pz")
e   = accessor.get_double_array("p0")
pdg = accessor.get_int_array("pdg")

y = 0.5 * np.log((e + pz) / (e - pz))
```

## Command-Line: brass-analyze

Run registered analyses over multiple run directories.

### Usage
```
brass-analyze [OPTIONS] OUTPUT_DIR ANALYSIS_NAME
```

### Arguments
- OUTPUT_DIR — top directory containing run subfolders (default pattern `out-*`)
- ANALYSIS_NAME — name of a registered analysis (see `--list-analyses`)

### Options
- `--list-analyses` — list registered analyses and exit
- `--pattern PATTERN` — glob for run folders (default: `out-*`)
- `--keys KEY1 KEY2 ...` — dotted config keys for labeling runs (last segment used as name)
  - example: `--keys Modi.Collider.Sqrtsnn General.Nevents`
- `--results-subdir DIR` — subdirectory to store results (default: `data`)
- `--strict-quantities` — fail if quantities differ across runs (default: warn and use first)
- `-v, --verbose` — print detailed information

## YAML Output

Analyses write human-readable YAML files (e.g., `bulk.yaml`) including metadata and result arrays.

Example:
```yaml
merge_key:
  sqrts: 17.3
  system: "PbPb"
smash_version: "SMASH-3.2-38-g5c9a7cbef"
n_events: 40
d2N_dpT_dy:
  pt_range: [0, 3]
  y_range: [-4, 4]
  pt_bins: 30
  y_bins: 30
  counts: ...
```

## Run Analyses from Python
```python
import os
import yaml
import brass as br

# --- example run directories ---
RUN_DIRS = [
    "runs/out-001",
    "runs/out-002",
]

def load_meta(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    coll = (cfg.get("Modi", {}) or {}).get("Collider", {}) or {}
    proj = (coll.get("Projectile", {}) or {}).get("Particles", {}) or {}
    targ = (coll.get("Target", {}) or {}).get("Particles", {}) or {}

    Zp, Np = int(proj.get(2212, 0)), int(proj.get(2112, 0))
    Zt, Nt = int(targ.get(2212, 0)), int(targ.get(2112, 0))

    def sym(Z, N):
        if (Z, N) == (82, 126): return "Pb"
        if (Z, N) == (1, 0):    return "p"
        return f"A{Z+N}"

    system = f"{sym(Zt, Nt)}{sym(Zp, Np)}"
    sqrts  = coll.get("Sqrtsnn", "unknown")
    return f"system={system},sqrts={sqrts}"

def main():
    file_and_meta = []
    used_quantities = None

    for d in RUN_DIRS:
        bin_path  = os.path.join(d, "particles_binary.bin")
        yaml_path = os.path.join(d, "config.yaml")
        if not (os.path.isfile(bin_path) and os.path.isfile(yaml_path)):
            print(f"[skip] Missing files in {d}")
            continue

        # Optional: read quantities once (if present). Otherwise, use [].
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        q = (((cfg.get("Output", {}) or {}).get("Particles", {}) or {})
             .get("Quantities", []) or [])
        q = [str(x) for x in q]

        if used_quantities is None:
            used_quantities = q
        elif used_quantities != q:
            print(f"[warn] Quantities differ in {yaml_path}; using the first set.")

        meta = load_meta(yaml_path)
        file_and_meta.append((bin_path, meta))

    if not file_and_meta:
        raise SystemExit("No valid runs found.")

    br.run_analysis(
        file_and_meta=file_and_meta,      # [(path_to_bin, "meta=..."), ...]
        analysis_name="my_analysis",      # registered analysis name
        quantities=used_quantities or [], # [] if not specified in YAML
        save_output=True,
        print_output=False,
        output_folder="results"
    )
    print("[done] brass analysis finished -> results/")

if __name__ == "__main__":
    main()
```
## Performance

<img width="856" height="357" alt="image" src="https://github.com/user-attachments/assets/03b56538-1b2c-4bea-a3a9-8dd6922975de" />
