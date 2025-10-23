# BRASS (Binary Reader and Analysis Suite Software) [![DOI](https://zenodo.org/badge/999239823.svg)](https://doi.org/10.5281/zenodo.17426761)


A simple and extensible C++/Python library for reading and analyzing binary particle output files.

## Features
- Blazingly fast (see performance) 
- C++ binary file reader for particle data
- Plugin-style extensible analysis system (via registry macros)
- Histogramming utilities
- Developed primarily for Binary format used by SMASH (see https://theory.gsi.de/~smash/userguide/current/doxypage_output_binary.html)

## Performance
<img width="800" alt="performance plot" src="https://github.com/user-attachments/assets/79095661-4f6e-4762-9b0e-f16df368dd28" />

## Build Instructions
in repository
```bash
pip install .
```

or from PyPI


```bash
pip install pybrass
```

## Simplest Usage

```py 

from brass import BinaryReader, Accessor

QUANTITIES = ["p0", "px", "py", "pz", "pdg"]

class Example(Accessor):
    def on_particle_block(self, block):
        arrays = dict(self.gather_block_arrays(block, QUANTITIES))
        E = arrays["p0"]
        px = arrays["px"]
        py = arrays["py"]
        pz = arrays["pz"]
        pdg = arrays["pdg"]
        # do something with E, px, py, pz, pdg here

reader = BinaryReader("events.bin", QUANTITIES, Example())
reader.read()
```

# brass-analyze

Command-line tool for running registered analyses on multiple SMASH run directories.

## Usage

brass-analyze [OPTIONS] OUTPUT_DIR ANALYSIS_NAME

- OUTPUT_DIR — top directory containing run subfolders (`out-*` by default)
- ANALYSIS_NAME — name of a registered analysis (see `--list-analyses`)

## Options

--list-analyses
  List registered analyses and exit.

--pattern PATTERN
  Glob for run folders (default: out-*).

--keys KEY1 KEY2 ...
  Dotted keys from config for labeling runs (last segment used as name).
  Example:
    --keys Modi.Collider.Sqrtsnn General.Nevents

--results-subdir DIR
  Subdirectory to store results (default: data).

--strict-quantities
  Fail if Quantities differ across runs (default: warn and use first).

--load 
  Load python files containing an analysis class registration 

-v, --verbose
  Print detailed information.

## Writing Analyses

```python
import numpy as np
import brass as br
from pathlib import Path

class Dndydpt:
    def __init__(self, y_edges, pt_edges, track_pdgs=None):
        self.y_edges  = np.asarray(y_edges)
        self.pt_edges = np.asarray(pt_edges)

        # 2D histogram over (pt, y), with variance tracking for errors
        self.incl     = br.HistND([self.pt_edges, self.y_edges], track_variance=True)
        self.per_pdg  = {}   # pdg -> HistND([pt, y], track_variance=True)
        self.track    = set(track_pdgs or [])
        self.n_events = 0

    def on_particle_block(self, block, accessor, opts):
        self.n_events += 1
        pairs = accessor.gather_block_arrays(block, ["p0","pz","px","py","pdg"])
        cols  = {k: v for k, v in pairs}
        E, pz, px, py, pdg = cols["p0"], cols["pz"], cols["px"], cols["py"], cols["pdg"]

        m = (E > np.abs(pz))  # avoid invalid rapidity
        if not m.any():
            return
        E, pz, px, py, pdg = E[m], pz[m], px[m], py[m], pdg[m]

        y  = 0.5*np.log((E + pz) / (E - pz))
        pt = np.hypot(px, py)

        # inclusive fill
        self.incl.fill(pt, y)

        # optional per-PDG fills
        if self.track:
            pdgs_here = np.intersect1d(np.unique(pdg), np.fromiter(self.track, dtype=int))
            for val in pdgs_here:
                sel = (pdg == val)
                H = self.per_pdg.setdefault(
                    int(val), br.HistND([self.pt_edges, self.y_edges], track_variance=True)
                )
                H.fill(pt, y, mask=sel)

    def merge_from(self, other, opts):
        self.incl.merge_(other.incl)
        for k, H in other.per_pdg.items():
            self.per_pdg.setdefault(k, br.HistND([self.pt_edges, self.y_edges], track_variance=True))
            self.per_pdg[k].merge_(H)
        self.n_events += getattr(other, "n_events", 0)

    def finalize(self, opts):
        """
        Normalize to d^2N/(dy dpt) per event, and compute per-bin errors.
        Error per bin = sqrt(variance), with variance propagated under normalization.
        """
        dy  = np.diff(self.y_edges)   # (ny,)
        dpt = np.diff(self.pt_edges)  # (npt,)
        area = dpt[:, None] * dy[None, :]     # (npt, ny), supports non-uniform bins
        n_ev = max(int(self.n_events), 1)

        # Normalize counts -> per-event per (dy dpt)
        self._normalize_hist2d_(self.incl, n_ev, area)
        for H in self.per_pdg.values():
            self._normalize_hist2d_(H, n_ev, area)

    @staticmethod
    def _normalize_hist2d_(H, n_ev, area):
        """
        Counts and variance normalization:
          value_norm = counts / (n_ev * area)
          var_norm   = sumw2 / (n_ev^2 * area^2)   (since errors() = sqrt(sumw2))
        """
        H.counts[:] = H.counts / (n_ev * area)
        if getattr(H, "sumw2", None) is not None:
            H.sumw2[:] = H.sumw2 / ((n_ev * area) ** 2)

    def _fmt_val(self, v):
        return f"{round(v, 3):g}" if isinstance(v, float) else str(v)

    def _label_from_keys(self, keys: dict) -> str:
        parts = [f"{k}-{self._fmt_val(keys[k])}" for k in sorted(keys)]
        return "_".join(parts).replace("/", "-")

    def save(self, out_dir, keys, opts):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = self._label_from_keys(keys)
        path = out_dir / f"dndydpt_{tag}.npz"

        # Build dict of per-PDG outputs: values + errors (if available)
        pdg_arrays = {}
        for k, H in self.per_pdg.items():
            pdg_arrays[f"pdg_{k}"] = H.counts
            if getattr(H, "sumw2", None) is not None:
                pdg_arrays[f"pdg_{k}_err"] = np.sqrt(H.sumw2)

        # Inclusive arrays: values + errors
        extras = {}
        if getattr(self.incl, "sumw2", None) is not None:
            extras["H_inclusive_err"] = np.sqrt(self.incl.sumw2)

        np.savez(
            path,
            H_inclusive=self.incl.counts,
            y_edges=self.y_edges,
            pt_edges=self.pt_edges,
            n_events=int(self.n_events),
            **pdg_arrays,
            **extras,
            keys=keys,
            analysis_name=getattr(self, "name", "dndydpt_python"),
            version=getattr(self, "smash_version", None),
        )

# Register
edges_y  = np.linspace(-4, 4, 31)
edges_pt = np.linspace(0.0, 3.0, 31)

br.register_python_analysis(
    "dndydpt_py",
    lambda: Dndydpt(edges_y, edges_pt, track_pdgs=[2212, 211, -211]),
    {},
)
```
## How Analyses Work

Each analysis plugin in BRASS subclasses the `Analysis` interface and is responsible for processing particle blocks and storing results.  

## Run an Analysis 

```python
import sys
import os
import argparse
import brass as br
import time
# 1) import your python analysis module so it registers itself
import dndydpt  

# 2) Quantities must EXACTLY match what the file contains
QUANTITIES = [
    "t","x","y","z",
    "mass","p0","px","py","pz",
    "pdg","id","charge","ncoll",
    "form_time","xsecfac",
    "proc_id_origin","proc_type_origin","time_last_coll",
    "pdg_mother1","pdg_mother2",
    "baryon_number","strangeness"
]
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/particles_oscar2013_extended.bin [outdir]")
        sys.exit(1)

    binfile = sys.argv[1]
    outdir  = sys.argv[2] if len(sys.argv) > 2 else "results_py"


    t0 = time.perf_counter()
    print(br.list_analyses())
    br.run_analysis(
        file_and_meta=[(binfile, "meta_key=1")],          
        analysis_names=["dndydpt_py"],          
        quantities=QUANTITIES,
        output_folder=outdir,
    )
    t1 = time.perf_counter()
    print(f"[PY] dndydpt_py elapsed: {t1-t0:.6f} s")

if __name__ == "__main__":
    main()

```


### Merging by Metadata

When you run over multiple binary files, BRASS uses user-supplied metadata (like `sqrt_s`, `projectile`, `target`) to associate results with a **merge key**. 
You define metadata like this:

```python
 br.run_analysis(
        file_and_meta=[(binfile_A, "meta_key=1"),(binfile_B, "meta_key=1"),(binfile_C, "meta_key=2")],          
        analysis_names=["dndydpt_py"],          
        quantities=QUANTITIES,
        output_folder=outdir,
    )
```
This will call the ``merge_from``method in ``Analysis`` class such that ``binfile_A``and ``binfile_B``will be merged. 


