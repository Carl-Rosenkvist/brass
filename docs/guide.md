# BRASS user guide: analyses and command-line tools

This guide shows how to go from SMASH runs to results with the BRASS
command-line tools, and how to write your own analysis.

- [The workflow](#the-workflow)
- [Preparing SMASH runs](#preparing-smash-runs)
- [Running analyses: `brass`](#running-analyses-brass)
  - [`bulk`](#bulk-rapidity-and-transverse-mass-spectra)
  - [`bulk-observables`](#bulk-observables-multiplicities-yields-mean-pt-and-mt)
  - [`centrality`](#centrality-impact-parameter-and-multiplicity)
  - [`collision-counts`](#collision-counts-interactions-by-process-type)
  - [`reaction-counts`](#reaction-counts-detailed-balance-of-reactions)
  - [`multiplicity-vs-time`](#multiplicity-vs-time-multiplicities-at-every-output-time)
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
section must use the binary format with an explicit `Quantities` list. If
several outputs have one (e.g. `Particles` and `Collisions`), the list of the
output whose file the analysis reads is used (`particles_*.bin` →
`Output.Particles`, `collisions_*.bin` → `Output.Collisions`). For example:

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

### `baryon-stopping`: x_F and rapidity spectra with ⟨pT⟩ and ⟨mT⟩

For each requested PDG code, fills one-dimensional histograms of the number
of particles and of their summed pT in Feynman x_F = 2 p_z/√s, and of the
number of particles and of their summed mT in rapidity. dN/dx_F, ⟨pT⟩(x_F),
dN/dy and ⟨mT⟩(y) follow exactly, without bins in pT or mT.

- **Reads:** `particles_binary.bin` or `particles_custom.bin`
- **Needs quantities:** `p0`, `px`, `py`, `pz`, `pdg`
- **Needs** `Modi.Collider.Sqrtsnn` and the centre-of-mass frame
  (`Calculation_Frame: center of mass`, or a symmetric system). The p_z axis
  spans −√s/2 … √s/2 of each run, so the x_F bins are exact and the same at
  every energy.
- **Skips** elastic events, like `bulk`.

| Option | Default | Meaning |
|---|---|---|
| `--pdgs PDG...` | required | PDG codes to histogram |
| `--add-pdgs PDG...` | none | Add species together, as for `bulk` |
| `--xf-bins N` | `80` | x_F bins in −1 < x_F < 1 |
| `--y-range MIN MAX` | `-4 4` | Rapidity range |
| `--y-bins N` | `40` | Rapidity bins |

```bash
brass baryon-stopping runs/* --pdgs 2212 -2212 211 3122 --add-pdgs 3122 3212 -o baryon-stopping.pkl
```

Each result has `histograms[kind][pdg]` for the kinds `"xf"` (particles),
`"xf_pt"` (summed pT), `"y"` (particles) and `"y_mt"` (summed mT), plus
`sqrts`, `events` and `pdgs`.

Do not derive x_F observables from the `bulk` (y, mT) histogram: with its
default bins dN/dx_F is off by 10–30 % and ⟨pT⟩(x_F) is biased by about as
much as the string parameters change it.

### `bulk-observables`: multiplicities, yields, mean pT and mT

The observables of the SMASH analysis suite's energy scan, for each
requested PDG code, from one pass of the histogram engine:

- 4π multiplicity and dN/dy at midrapidity (|y| < 0.5),
- ⟨pT⟩ and ⟨mT − m⟩ at midrapidity, exact (from pT- and mT − m-weighted
  histograms, not from bin centres),
- dN/dy, and d²N/(dpT dy) and d²N/(d(mT − m) dy) at midrapidity.

Outer bins catch everything outside the plotted ranges, so the totals are
exact. `mt0` (= mT − m) is a computed quantity that any histogram request can
use, like `pt` and `mt`.

- **Reads:** `particles_binary.bin` or `particles_custom.bin`
- **Needs quantities:** `p0`, `px`, `py`, `pz`, `pdg` (`mass` is used if present)
- **Counts every event**, including elastic ones (as the analysis suite did);
  `--skip-elastic` skips and does not count events with exactly two particles.

| Option | Default | Meaning |
|---|---|---|
| `--pdgs PDG...` | required | PDG codes to analyze |
| `--add-pdgs PDG...` | none | Add species, as for `bulk` |
| `--y-range MIN MAX`, `--y-bins N` | `-4 4`, `40` | Rapidity spectrum |
| `--midrapidity Y` | `0.5` | Midrapidity is \|y\| < Y |
| `--pt-edges E...`, `--mt0-edges E...` | 0 … 3.2 GeV | Bin edges of the midrapidity spectra |
| `--skip-elastic` | off | Skip two-particle events |

```bash
brass bulk-observables runs/* --pdgs 211 -211 321 -321 2212 -2212 3122 3212 -o bulk-observables.pkl
```

Each result has `observables(pdg)` (or `observables([pdg1, pdg2])` to add
species), a dictionary with `total_multiplicity`, `midrapidity_yield`,
`meanpt`, `meanmt`, and `yspectra`, `ptspectra`, `mtspectra` as
`(bin centres, values)`, all per event; `events` is the number of events.

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

### `collision-counts`: interactions by process type

Counts the interactions of the collision output by (process type, number of
incoming, number of outgoing particles), e.g. to compare the scattering rate
of a box with the expected one.

- **Reads:** `collisions_binary.bin` or `collisions_custom.bin`
- **Needs quantities:** `t` (in `Output.Collisions`)

| Option | Default | Meaning |
|---|---|---|
| `--t-start T` | none | Count only interactions after time T (fm), the time of the first incoming particle |

```bash
brass collision-counts runs/* --t-start 5 -o collision-counts.pkl
```

Each result has `counts_for(process=..., n_in=..., n_out=...)` (any argument
can be left out) and `events`. The process types are SMASH's: 1 elastic,
2 2→1, 3 2→2, 5 decay, 6 wall crossing, 41–49 strings, ...

### `reaction-counts`: detailed balance of reactions

Counts every reaction of the collision output, binned in √s of the incoming
particles (or in |t|), so that each reaction can be compared with its inverse
in a box in equilibrium. A reaction is (sorted incoming PDGs, sorted outgoing
PDGs): a decay ρ⁰ → π⁺π⁻ is the inverse of the formation π⁺π⁻ → ρ⁰.

- **Reads:** `collisions_binary.bin` or `collisions_custom.bin`
- **Needs quantities:** `t`, `p0`, `px`, `py`, `pz`, `pdg` (in `Output.Collisions`)
- **Skips** 1 → 1 interactions (wall crossings) and interactions before `--t-start`.

| Option | Default | Meaning |
|---|---|---|
| `--t-start T` | none | Count only after time T (fm), e.g. once the box is in equilibrium |
| `--variable sqrts\|t` | `sqrts` | Bin in √s, or in \|t\| = \|(p_in[0] − p_out[0])²\| (2 → 2 only) |
| `--range MIN MAX`, `--bins N` | 0–5 GeV, 100 (√s); 0–30 GeV², 60 (t) | Binning |

```bash
brass reaction-counts runs/* --t-start 20 -o reaction-counts.pkl
```

Each result has `histogram(reaction)` (counts per bin), `pairs()` (each
reaction with its inverse), `rate(reaction)` (per event, per fm after
`t_start` and per bin width, with `End_Time` from the config),
`most_violating(n)` (the n pairs with the largest |√N_f − √N_b|), `edges`
and `events`.

### `multiplicity-vs-time`: multiplicities at every output time

Mean number of particles of each PDG code at every output time, with its
statistical error, e.g. to see when a box reaches equilibrium.

- **Reads:** `particles_binary.bin` or `particles_custom.bin`, written at
  several times per event (`Only_Final: No`)
- **Needs quantities:** `t`, `pdg`

| Option | Default | Meaning |
|---|---|---|
| `--pdgs PDG...` | required | PDG codes to count |

Each result has `times`, `mean(pdg)` and `error(pdg)` (√(variance / (events − 1)))
and `events`. Counts include test particles.

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

## Experimental data: `brass.expdata`

The `exp_data/` folder of the brass source tree holds measured spectra and
yields as CSV files with `# key: value` header lines, sorted by system
(`pp/`, `AA/`). `brass.expdata` reads them:

```python
from brass import expdata

data = expdata.find("multiplicity", "pp", 211)   # exp_data/pp/multiplicity_211.csv
sqrts, mult, err = data.points("sqrtsnn", "multiplicity")
print(data.meta, data.text_columns["source"][:3])
```

`points` returns the total uncertainty: a `<value>_err` column, or
statistical and systematic errors added in quadrature. Set `BRASS_EXP_DATA`
to read another folder in the same format (the data is not part of the
wheel). The energy-scan files (`multiplicity_`, `midy_`, `meanpt_`,
`meanmt_<pdg>.csv`) were converted from the data collection of the SMASH
analysis suite; each row keeps its reference in the `source` column. Read
`exp_data/README.md` before using them.

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

## Calibrating model parameters: `brass.emulation`

With `bulk` results from a parameter scan (for example generated with
`brass-scan`), BRASS can build an emulator — a fast surrogate for SMASH — and
compare it to experimental data. It needs the `emulation` extra
(`pip install "pybrass[emulation]"`).

**1. Fit the emulator.** `parameter_paths` says where each parameter is found
in the run's `config.yaml`. A string key inside a list of PYTHIA settings
(`"key = value"` entries) is looked up by name. Every parameter is a GP
input, including the collision energy, so one emulator covers several
energies.

```python
from brass.emulation.bulk import BulkEmulator

paths = {
    "Sqrtsnn": ("Modi", "Collider", "Sqrtsnn"),
    "Strange_Supp": ("Collision_Term", "String_Parameters", "Strange_Supp"),
    "probSQtoQQ": ("Collision_Term", "String_Parameters", "Pythia_Settings",
                   "StringFlav:probSQtoQQ"),
}

emu = BulkEmulator.from_file("bulk.pkl", paths, observables=["dNdy"])
emu.fit(verbose=True)          # prints the PCA components per particle
emu.save("emulator.npz")       # later: emu = BulkEmulator.load("emulator.npz")
```

- `observables`: `"dNdy"` (default), `"dNdmt"` or `"hist2d"`. Emulate only what
  you compare to data: the 2D histogram of a rare particle is mostly
  statistical noise and needs many more components.
- `n_components`: a fraction of variance to keep (default `0.99`) or a number.
- `log_parameters=["Sqrtsnn"]`: use log(√s) as GP input, which suits energy
  scans.
- `mt_max=3.0, mt_group=2`: before fitting, drop mT bins above 3 GeV and merge
  pairs of neighbouring mT bins (particle numbers are kept). This makes a
  `"hist2d"` emulator several times smaller. Particles above `mt_max` are
  lost, also from the dN/dy derived from it.

From a `"hist2d"` emulator, `MtSpectrum(y_min, y_max)` (d²N/(dy dmT)
averaged over a rapidity window) is derived as well. Choose window edges on
the y bin edges; partly covered bins are weighted by the overlap.

**x_F observables** come from `baryon-stopping` results and their own
emulator, with the same parameters:

```python
from brass.emulation.baryon_stopping import BaryonStoppingEmulator

bs = BaryonStoppingEmulator.from_file(
    "baryon-stopping.pkl", paths, observables=["dNdxF", "meanpT_xF"])
bs.fit()
xf = bs.axis(2212, "meanpT_xF")                 # x_F bin centres
mean, std = bs.predict_observable(theta, 2212, "meanpT_xF", return_std=True)
```

- Observables: `"dNdxF"`, `"meanpT_xF"`, `"dNdy"` and `"meanmT_y"` (⟨mT⟩, so
  subtract the mass for ⟨mT⟩ − m), each in the bins of the analysis.
- ⟨pT⟩ and ⟨mT⟩ are emulated directly. In a pp scan this reached the
  statistical precision of the runs with 3–5 PCA components; deriving them
  from emulated spectra needed far more.
- They are emulated in the bins that hold particles in every training run,
  so for rare hadrons the x_F range is narrower; `axis()` gives the bins.

Rare particles have too few entries for ⟨pT⟩(x_F) far from x_F = 0, so choose
per particle which observables to emulate, over which x_F range and in which
bins (plot the training data first to see where it is noisy):

```python
bs = BaryonStoppingEmulator.from_file(
    "baryon-stopping.pkl", paths,
    observables={
        2212:  ["dNdxF", "meanpT_xF"],
        211:   ["dNdxF", "meanpT_xF"],
        3122:  ["meanpT_xF"],
        -2212: ["meanpT_xF"],
    },
    xf_max={211: 0.8, 3122: 0.8, -2212: 0.3},   # |x_F| range per particle
    xf_group=2,                                 # merge 2 x_F bins (0.05 wide)
)
```

Particles not in `observables` are not emulated. `xf_max` and `xf_group` are
a number for all particles or a dict per PDG; merging bins adds counts and
summed pT exactly. A choice that leaves no usable bin stops `fit()` with a
message naming the particle and observable.

Usually better than fixed bins and ranges is to let the statistics choose:

```python
bs = BaryonStoppingEmulator.from_file(
    "baryon-stopping.pkl", paths,
    observables={2212: ["dNdxF", "meanpT_xF"], 211: ["dNdxF", "meanpT_xF"]},
    fold=True,        # add -x_F and +x_F (pp is symmetric): emulate x_F >= 0
    min_count=500,    # merge bins outward from x_F = 0 to >= 500 particles per run
)
```

With `min_count`, bins stay at the analysis binning where statistics are
high and widen where they are low, so the training data are less noisy and
fewer PCA components (GP fits) are needed. For π⁺ in a pp scan this halved
the number of GP fits and kept the full x_F range. `bs.bins(pdg, observable)`
gives the bin edges; both settings also accept a dict per PDG and apply to
the rapidity observables as well.

**2. Check it.** `emu.validate(holdout=0.2)` refits without 20 % of the
parameter points and predicts them. The standardised residuals
`(prediction − simulation) / σ` should be roughly standard normal; much wider
means the emulator is overconfident.

**3. Build the likelihood** once, outside the MCMC loop:

```python
from brass.emulation import EmulatorLikelihood

like = EmulatorLikelihood(emu, rel_error_floor=0.03)
like.add(211, y_exp, dndy_exp, err_exp, fixed={"Sqrtsnn": 17.3})
like.add(211, y_exp2, dndy_exp2, err_exp2, fixed={"Sqrtsnn": 8.76})
print(like.parameters)         # the fitted parameters, in theta order
```

`observable=` selects what the data are, and `emulator=` takes the dataset
from another emulator of the same scan, so dN/dy and ⟨pT⟩(x_F) enter one
likelihood:

```python
like.add(2212, xf_exp, mean_pt_exp, err_exp, observable="meanpT_xF",
         emulator=bs, fixed={"Sqrtsnn": 17.3})
```

Do not add both dN/dy and dN/dx_F measured on the same particles without
accounting for their correlation.

**4. Sample.** `like(thetas)` takes all walkers at once and returns their
log-likelihoods, so it works directly with `emcee`'s `vectorize=True`:

```python
import emcee
import numpy as np

def log_probability(thetas):
    inside = np.all((thetas >= lower) & (thetas <= upper), axis=1)
    logp = np.full(len(thetas), -np.inf)
    if inside.any():
        logp[inside] = like(thetas[inside])
    return logp

sampler = emcee.EnsembleSampler(n_walkers, len(like.parameters),
                                log_probability, vectorize=True)
```

The likelihood is Gaussian with the experimental errors plus the emulator's
covariance (GP uncertainty and PCA truncation error, correlated between
points). `covariance="diagonal"` uses only its diagonal; `"none"` ignores it.
Run the chain long enough: check `sampler.get_autocorr_time()` and use many
times that number of steps.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `invalid int value: 'runs/run0'` | Run folders were placed after `--pdgs`. Put the folders first, or end the list with `--`. |
| `unrecognized arguments: --load` | `--load` must come before the analysis name: `brass --load file.py NAME ...` |
| `no matches found: ...=[0,3],[3,6]` | The shell expanded the brackets. Quote the `--param` value. |
| `No matching data file found in ...` | The folder has no file the analysis reads (see *Reads* for each analysis). |
| `Expected exactly one matching data file` | The folder has several matching files, e.g. both `particles_binary.bin` and `particles_custom.bin`. |
| `Expected exactly one Quantities section, found N` | The data file's output (`Output.Particles`, `Output.Dileptons` or `Output.Collisions`, chosen by the file name) has no `Quantities` list, or none of several lists matches the file. |
| `particle quantity not in file: charge` | The analysis needs a quantity the run did not store. Add it to `Quantities` in the SMASH config. |
| `cannot add histograms with different binning` | Results analyzed with different options were merged. Re-run with the same options. |
| `ModuleNotFoundError` when loading a pickle | The pickle contains your own analysis class; add its folder to `sys.path` first. |
| `BrokenProcessPool` when using `AnalysisRunner` in a script | Put the code that starts the runner inside `if __name__ == "__main__":`. |
