# BRASS user guide: analyses and command-line tools

This guide shows how to go from SMASH runs to results with the BRASS
command-line tools, and how to write your own analysis.

- [The workflow](#the-workflow)
- [Preparing SMASH runs](#preparing-smash-runs)
- [Running analyses: `brass`](#running-analyses-brass)
  - [`bulk`](#bulk-rapidity-and-transverse-mass-spectra)
  - [`centrality`](#centrality-impact-parameter-and-multiplicity)
  - [`dileptons`](#dileptons-invariant-mass-spectra)
- [Reading the results](#reading-the-results)
- [Combining results: `brass-merge`](#combining-results-brass-merge)
- [Generating SMASH commands: `brass-scan`](#generating-smash-commands-brass-scan)
- [Writing your own analysis](#writing-your-own-analysis)
- [Troubleshooting](#troubleshooting)

## The workflow

```text
brass-scan  ──►  SMASH runs  ──►  brass <analysis>  ──►  brass-merge  ──►  your Python
(commands)       (one folder      (one result per       (optional:        (plots, fits)
                  per run)         configuration)        combine jobs)
```

1. **`brass-scan`** prints one SMASH command per parameter point (and per job).
2. You run SMASH; each run writes its own output folder.
3. **`brass <analysis>`** reads many run folders in parallel and saves the
   results to a pickle file. Runs with the same physics configuration are
   added together automatically.
4. **`brass-merge`** combines pickle files from separate `brass` calls, for
   example from different cluster jobs.
5. You load the pickle in Python and work with the histograms.

## Preparing SMASH runs

Every run folder must contain:

- the **`config.yaml`** that SMASH writes into its output folder, and
- **exactly one** binary data file the analysis reads (see each analysis).

BRASS takes the list of stored quantities from the config, so the output
section must use the binary format with an explicit `Quantities` list, and
**only one** of `Output.Particles`, `Output.Dileptons` or `Output.Collisions`
may have a `Quantities` list. For example:

```yaml
Output:
  Particles:
    Format: [Binary]
    Quantities: [p0, px, py, pz, pdg, charge]
```

## Running analyses: `brass`

```bash
brass ANALYSIS RUN_FOLDERS... [options]
```

| Option | Meaning |
|---|---|
| `-o FILE`, `--output FILE` | Output pickle (default: `ANALYSIS.pkl`, e.g. `bulk.pkl`) |
| `-j N`, `--max-workers N` | Number of parallel processes (default: one per CPU core) |
| `--load FILE` | Add analyses from your own Python file (see [below](#writing-your-own-analysis)); goes **before** `ANALYSIS` |

`brass --help` lists the available analyses, and `brass ANALYSIS --help`
lists the options of one analysis.

> **Put the run folders first.** Options such as `--pdgs` take a list of
> values and would also swallow folder names placed after them:
>
> ```bash
> brass bulk runs/* --pdgs 211 2212         # works
> brass bulk --pdgs 211 2212 -- runs/*      # works: `--` ends the list
> brass bulk --pdgs 211 2212 runs/*         # error: invalid int value: 'runs/run0'
> ```

**Runs are grouped by configuration.** BRASS compares the `config.yaml` of
all folders, ignoring `General.Randomseed`, `Logging`, `Output`, `decaymodes`
and `particles`. Runs that are otherwise identical are added together into
one result; runs that differ (for example in `Sqrtsnn`) give separate
results. Passing 300 runs at three energies therefore gives three results:

```console
$ brass bulk runs/* --pdgs 211 2212
Saved 3 grouped result(s) to bulk.pkl
```

### `bulk`: rapidity and transverse-mass spectra

Fills a two-dimensional (y, mT) histogram for each requested PDG code.

- **Reads:** `particles_binary.bin` or `particles_custom.bin`
- **Needs quantities:** `p0`, `px`, `py`, `pz`, `pdg` (`mass` is used if present)
- **Skips** events with exactly two particles (elastic collisions); they are
  not counted in `events` either.

| Option | Default | Meaning |
|---|---|---|
| `--pdgs PDG...` | required | PDG codes to histogram |
| `--add-pdgs PDG...` | none | Add species together: the first PDG receives the others, e.g. `--add-pdgs 3122 3212` adds Σ⁰ to Λ. Repeat the option for several groups. The first PDG must be in `--pdgs`. |
| `--y-range MIN MAX` | `-4 4` | Rapidity range |
| `--y-bins N` | `50` | Rapidity bins |
| `--mt-range MIN MAX` | `0 10` | Transverse-mass range in GeV |
| `--mt-bins N` | `100` | Transverse-mass bins |

```bash
brass bulk runs/* --pdgs 211 -211 2212 3122 --add-pdgs 3122 3212 -o bulk.pkl
```

Each result has:

| Attribute | Content |
|---|---|
| `histograms[pdg]` | Histogram with `.values` (counts, shape `(y_bins, mt_bins)`) and `.edges` (`[y_edges, mt_edges]`) |
| `events` | Number of analyzed events |
| `pdgs` | The PDG codes in `histograms` |
| `config` | The SMASH configuration of the first run in the group |

### `centrality`: impact parameter and multiplicity

Records, for every event, the impact parameter and the number of charged
particles. Use it to define centrality classes.

- **Reads:** `particles_binary.bin` or `particles_custom.bin`
- **Needs quantities:** `charge`
- **No options**

```bash
brass centrality runs/* -o centrality.pkl
```

Each result has `impact_parameters` and `charged_multiplicities`, two lists
with one entry per event.

### `dileptons`: invariant-mass spectra

Histograms the invariant mass of lepton pairs from the dilepton output,
separately for each decay channel, weighted by the shining weight.

- **Reads:** `Dileptons_custom.bin`
- **Needs quantities:** `p0`, `px`, `py`, `pz`, `pdg` (in `Output.Dileptons`)

| Option | Default | Meaning |
|---|---|---|
| `--bins N` | `50` | Number of mass bins |
| `--mass-min M` | `0.0` | Lower mass in GeV |
| `--mass-max M` | `1.5` | Upper mass in GeV |
| `--pair-pdgs PDG1 PDG2` | `-11 11` | Lepton pair, e.g. `-13 13` for muons |
| `--weight-quantity NAME` | `weight` | Weight per decay (`weight` = the interaction's `sigma`) |

```bash
brass dileptons runs/* --bins 60 --mass-max 1.2 -o dileptons.pkl
```

Each result has `values[channel]` (summed weights per bin), `edges[channel]`
(bin edges) and `nevents`. Channels are named like `"113 -> -11 11"`
(ρ⁰ → e⁺e⁻).

## Reading the results

The output is a pickle file containing a list with one entry per
configuration:

```python
import pickle

with open("bulk.pkl", "rb") as f:
    results = pickle.load(f)

for entry in results:
    bulk = entry["result"]          # the analysis object, e.g. Bulk
    print(bulk.config["Modi"]["Collider"]["Sqrtsnn"], bulk.events)
```

Use `result.config` to look up parameters; it is the ordinary nested
dictionary from `config.yaml`. (`entry["meta"]` holds the same configuration
as nested tuples, which BRASS uses internally for grouping.)

The histograms hold raw counts. To get dN/dy and dN/dmT per event, divide by
the number of events and the bin widths:

```python
import numpy as np

hist = bulk.histograms[211]
counts = np.asarray(hist.values)               # shape (y_bins, mt_bins)
y_edges, mt_edges = (np.asarray(e) for e in hist.edges)

dN_dy = counts.sum(axis=1) / (bulk.events * np.diff(y_edges))
dN_dmt = counts.sum(axis=0) / (bulk.events * np.diff(mt_edges))
y = 0.5 * (y_edges[:-1] + y_edges[1:])
```

For dileptons, divide the summed weights by the number of events and the bin
width the same way:

```python
dl = results[0]["result"]
channel = "113 -> -11 11"
dN_dM = dl.values[channel] / (dl.nevents * np.diff(dl.edges[channel]))
```

## Combining results: `brass-merge`

When runs are analyzed in several `brass` calls, for example one per cluster
job, `brass-merge` combines the pickle files. Results with the same
configuration are added together, just as within one `brass` call:

```bash
brass bulk job1/runs/* --pdgs 211 -o part1.pkl
brass bulk job2/runs/* --pdgs 211 -o part2.pkl
brass-merge part1.pkl part2.pkl -o bulk.pkl
```

Use the **same analysis options** (bins, ranges, PDG codes) for every part;
results with different binning cannot be added and `brass-merge` stops with
`cannot add histograms with different binning`.

## Generating SMASH commands: `brass-scan`

`brass-scan` prints one SMASH command for every combination of parameter
values. Each `--param` gives a configuration key (dot-separated, as in
`config.yaml`) and comma-separated values:

```bash
brass-scan --prefix "smash -i config.yaml" \
  --param Modi.Collider.Sqrtsnn=8.8,17.3 \
  --param 'Modi.Collider.Impact.Range=[0,3],[3,6]'
```

```text
smash -i config.yaml -c '{"Modi":{"Collider":{"Sqrtsnn":8.8,"Impact":{"Range":[0,3]}}}}'
smash -i config.yaml -c '{"Modi":{"Collider":{"Sqrtsnn":8.8,"Impact":{"Range":[3,6]}}}}'
smash -i config.yaml -c '{"Modi":{"Collider":{"Sqrtsnn":17.3,"Impact":{"Range":[0,3]}}}}'
smash -i config.yaml -c '{"Modi":{"Collider":{"Sqrtsnn":17.3,"Impact":{"Range":[3,6]}}}}'
```

- Values are read as JSON: numbers, lists (`[0,3]`) and objects
  (`{"2212":1}`) keep their type; anything else, like `Yes`, becomes a string.
- **Quote values that contain brackets or braces.** Shells such as zsh treat
  `[0,3]` as a file pattern and stop with `no matches found`.

To split each parameter point into several jobs, give the events per job and
the total number of events. Every job gets `General.Nevents` set:

```bash
brass-scan --prefix smash --param Modi.Collider.Sqrtsnn=8.8,17.3 \
  --events-per-job 500 --max-events 1500
```

This prints three identical commands per energy (1500 / 500). SMASH chooses a
new random seed for each run unless the config fixes one, so the jobs are
independent; `brass` later adds them together.

`brass-scan` only prints commands. Give every run its **own output folder**,
for example:

```bash
brass-scan --prefix "smash -i config.yaml" --param Modi.Collider.Sqrtsnn=8.8,17.3 \
  --events-per-job 500 --max-events 1500 > commands.txt

i=0
while read -r cmd; do
  eval "$cmd -o runs/run$i" &
  i=$((i + 1))
done < commands.txt
wait
```

This starts all runs at once. For many runs, submit the lines of
`commands.txt` to your cluster's batch system instead.

## Writing your own analysis

An analysis is a Python class derived from `brass.Analysis`. BRASS takes care
of finding the config and data file in each run folder, running folders in
parallel, grouping by configuration, and saving the results.

This example computes the mean transverse momentum of one particle species:

```python
# mean_pt.py
from pathlib import Path

import numpy as np

import brass


class MeanPt(brass.Analysis):
    """Mean transverse momentum of one particle species."""

    @classmethod
    def add_arguments(cls, parser):
        # Command-line options of this analysis (optional).
        parser.add_argument("--pdg", type=int, default=211, help="PDG code")

    def __init__(self, folder_path, args):
        super().__init__(folder_path, args)  # reads config.yaml
        self.pdg = args.pdg
        self.sum_pt = 0.0
        self.count = 0

    def data_file_matches(self, path: Path) -> bool:
        # Which file in the run folder to read (exactly one must match).
        return path.name in ("particles_binary.bin", "particles_custom.bin")

    def analyze_file(self, file_path):
        # Called once per run folder. self.quantities comes from config.yaml.
        reader = brass.BinaryReader(str(file_path), self.quantities)

        while (block := reader.read()) is not None:
            if isinstance(block, brass.ParticleBlock):
                particles = block.particles
                selected = particles.column("pdg") == self.pdg
                pt = np.hypot(particles.column("px"), particles.column("py"))
                self.sum_pt += pt[selected].sum()
                self.count += int(selected.sum())

    def __iadd__(self, other):
        # Add another run with the same configuration into this one.
        self.sum_pt += other.sum_pt
        self.count += other.count
        return self

    def mean_pt(self):
        return self.sum_pt / self.count
```

Run it by loading the file; the class name becomes the analysis name:

```bash
brass --load mean_pt.py MeanPt runs/* --pdg 2212 -o mean_pt.pkl
```

The four methods to provide are `data_file_matches`, `analyze_file` and
`__iadd__` (required), and `add_arguments` (only if the analysis has
options). Inside `analyze_file` you can use everything in `brass`: reading
blocks, NumPy columns, or the histogram engine (`brass.histogram`,
see the [README](../README.md)).

**Reading the results back** needs the analysis class, so Python must be able
to import your file:

```python
import pickle
import sys

sys.path.insert(0, "path/to/folder/with/mean_pt.py")

with open("mean_pt.pkl", "rb") as f:
    for entry in pickle.load(f):
        print(entry["result"].mean_pt())
```

Without that, loading fails with `ModuleNotFoundError: No module named 'mean_pt'`.

**From Python** instead of the command line, `AnalysisRunner` does the same as
`brass`:

```python
# run_mean_pt.py
import argparse

from brass import AnalysisRunner
from mean_pt import MeanPt

if __name__ == "__main__":
    args = argparse.Namespace(pdg=2212)
    runner = AnalysisRunner(MeanPt, ["runs/run0", "runs/run1"], args=args)
    results = runner.run()            # or runner.run_and_save("mean_pt.pkl")
```

`AnalysisRunner` runs the folders in parallel processes, so in a script the
code that starts it must be inside `if __name__ == "__main__":`. Without it,
the run fails on macOS and Windows with `BrokenProcessPool`. Pass
`max_workers=N` to limit the number of processes.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `invalid int value: 'runs/run0'` | Run folders were placed after `--pdgs`. Put the folders first, or end the list with `--`. |
| `unrecognized arguments: --load` | `--load` must come before the analysis name: `brass --load file.py NAME ...` |
| `no matches found: ...=[0,3],[3,6]` | The shell expanded the brackets. Quote the `--param` value. |
| `No matching data file found in ...` | The folder has no file the analysis reads (see *Reads* for each analysis). |
| `Expected exactly one matching data file` | The folder has several matching files, e.g. both `particles_binary.bin` and `particles_custom.bin`. |
| `Expected exactly one Quantities section, found N` | `config.yaml` must have a `Quantities` list in exactly one of `Output.Particles`, `Output.Dileptons` and `Output.Collisions`. |
| `particle quantity not in file: charge` | The analysis needs a quantity the run did not store. Add it to `Quantities` in the SMASH config. |
| `cannot add histograms with different binning` | Results analyzed with different options were merged. Re-run with the same options. |
| `ModuleNotFoundError` when loading a pickle | The pickle contains your own analysis class; add its folder to `sys.path` first. |
| `BrokenProcessPool` when using `AnalysisRunner` in a script | Put the code that starts the runner inside `if __name__ == "__main__":`. |
