# BRASS
Binary Reader and Analysis Suite
A simple and extensible C++/Python library for reading and analyzing binary particle output files.

## Features
- C++ header-based binary file reader for particle data
- Plugin-style extensible analysis system (via registry macros)
- Optional Python bindings via `pybind11`
- Histogramming utilities
- No external dependencies (except optionally `pybind11`)
- Devloped primarly for SMASH
## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

This reads the binary file and dispatches particle blocks to any registered analysis.

## Writing Analyses

You can create a custom analysis by subclassing `Analysis`:

```cpp
class MyAnalysis : public Analysis {
public:
    void analyze_particle_block(const ParticleBlock& block, const Accessor& accessor) override {
        // your logic here
    }
    void finalize() override {}
    void save(const std::string& path) override {}
};

REGISTER_ANALYSIS("my_analysis", MyAnalysis)
```

```cpp
#include "analysis.h"
#include "analysisregister.h"
#include "histogram2d.h"
#include <cmath>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <string>

class BulkObservables : public Analysis {
public:
    BulkObservables()
      : y_min(-4.0), y_max(4.0), y_bins(30),
        pt_min(0.0), pt_max(3.0), pt_bins(30),
        d2N_dpT_dy(pt_min, pt_max, pt_bins,
                   y_min, y_max, y_bins),
        n_events(0) {}

    Analysis& operator+=(const Analysis& other) override {
        auto const* o = dynamic_cast<const BulkObservables*>(&other);
        if (!o) throw std::runtime_error("merge mismatch");

        d2N_dpT_dy += o->d2N_dpT_dy;
        n_events   += o->n_events;

        for (auto const& [pdg, src] : o->per_pdg_) {
            auto& dst = obs_for(pdg);
            dst.d2N_dpT_dy += src.d2N_dpT_dy;
        }
        return *this;
    }

    void analyze_particle_block(const ParticleBlock& b, const Accessor& a) override {
        // event selection: needs at least one wounded nucleon
        bool has_wounded = false;
        for (size_t i = 0; i < b.npart; ++i) {
            int pdg = a.get_int("pdg", b, i);
            if ((pdg == 2212 || pdg == 2112) && a.get_int("ncoll", b, i) > 0) {
                has_wounded = true;
                break;
            }
        }
        if (!has_wounded) return;
        ++n_events;

        // fill spectra
        for (size_t i = 0; i < b.npart; ++i) {
            const int pdg   = a.get_int("pdg", b, i);
            const double E  = a.get_double("p0", b, i);
            const double pz = a.get_double("pz", b, i);
            const double px = a.get_double("px", b, i);
            const double py = a.get_double("py", b, i);

            if (E <= std::abs(pz)) continue;
            const double y  = 0.5 * std::log((E + pz) / (E - pz));
            const double pt = std::hypot(px, py);

            d2N_dpT_dy.fill(pt, y);

            auto& obs = obs_for(pdg);
            obs.d2N_dpT_dy.fill(pt, y);
        }
    }

    void finalize() override {
        if (n_events > 0) {
              d2N_dpT_dy.scale(1.0 / n_events);
              for (auto& [_, o] : per_pdg_) {
                 o.d2N_dpT_dy.scale(1.0 / n_events);
             }
         }
    }

    void save(const std::string& dir) override {
        YAML::Emitter out;
        out << YAML::BeginMap;

        out << YAML::Key << "merge_key"     << YAML::Value; to_yaml(out, keys);
        out << YAML::Key << "smash_version" << YAML::Value << YAML::DoubleQuoted << smash_version;
        out << YAML::Key << "n_events"      << YAML::Value << n_events;

        // totals
        out << YAML::Key << "d2N_dpT_dy"    << YAML::Value;
        to_yaml(out, "pt", "y", d2N_dpT_dy);

        // per-PDG spectra
        out << YAML::Key << "spectra" << YAML::Value << YAML::BeginMap;
        for (auto const& [pdg, o] : per_pdg_) {
            out << YAML::Key << std::to_string(pdg) << YAML::Value;
            to_yaml(out, "pt", "y", o.d2N_dpT_dy);
        }
        out << YAML::EndMap;

        out << YAML::EndMap;

        std::ofstream f(dir + "/bulk.yaml");
        f << out.c_str();
    }

private:
    struct Obs {
        Histogram2D d2N_dpT_dy;
        Obs(double pt_min, double pt_max, size_t pt_bins,
            double y_min, double y_max, size_t y_bins)
          : d2N_dpT_dy(pt_min, pt_max, pt_bins,
                       y_min, y_max, y_bins) {}
    };

    Obs& obs_for(int pdg) {
        auto [it, inserted] = per_pdg_.try_emplace(
            pdg, pt_min, pt_max, pt_bins, y_min, y_max, y_bins);
        return it->second;
    }

    double y_min, y_max, pt_min, pt_max;
    size_t y_bins, pt_bins;
    Histogram2D d2N_dpT_dy;
    int n_events;

    std::unordered_map<int, Obs> per_pdg_;
};

REGISTER_ANALYSIS("BulkObservables", BulkObservables);

```
## How Analyses Work

Each analysis plugin in BRASS subclasses the `Analysis` interface and is responsible for processing particle blocks and storing results.

### Merging by Metadata

When you run over multiple binary files, BRASS uses user-supplied metadata (like `sqrt_s`, `projectile`, `target`) to associate results with a **merge key**. 
You define metadata like this:

```bash
./binary_reader file1.bin:sqrt_s=5.02,target=Pb file2.bin:sqrt_s=5.02,target=Pb simple pdg pz p0
```

### YAML Output

Each analysis writes a human-readable YAML file named after the analysis, e.g., `simple.yaml`, which contains:
- The `smash_version`
- The `merge_keys`
- The `data` block with all computed quantities

Example output:
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
## Python usage
```py 
from brass import BinaryReader, CollectorAccessor
import numpy as np
accessor = CollectorAccessor()
reader = BinaryReader("particles_binary.bin", ["pdg", "pz", "p0"], accessor)
reader.read()

pz = accessor.get_double_array("pz")
e  = accessor.get_double_array("p0")
pdg = accessor.get_int_array("pdg")

y = 0.5 * np.log((e + pz) / (e - pz))
```

## Run analyses through Python
```py 

import os
import yaml
import brass as br

# --- point these to a few run directories you want to analyze ---
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
        # Just a couple of common cases; falls back to A=Z+N
        if   (Z, N) == (82, 126): return "Pb"
        if   (Z, N) == (1, 0):    return "p"
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
        file_and_meta=file_and_meta,     # [(path_to_bin, "meta=..."), ...]
        analysis_name="my_analysis",     # label for outputs
        quantities=used_quantities or [],# [] if you don't care / not in YAML
        save_output=True,
        print_output=False,
        output_folder="results"          # will be created if missing
    )
    print("[done] brass analysis finished -> results/")

if __name__ == "__main__":
    main()
```
