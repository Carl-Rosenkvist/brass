# BRASS (Binary Reader and Analysis Suite Software) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17426761.svg)](https://doi.org/10.5281/zenodo.17426761)


**Binary Reader and Analysis Suite Software**

BRASS is a small C++/Python library for reading and analyzing SMASH binary particle output files.

It provides:

- a fast C++ binary reader
- zero-copy NumPy access to particle columns
- C++ histogramming powered by Boost.Histogram: https://github.com/boostorg/histogram
- one-dimensional, two-dimensional, and N-dimensional histograms
- multiple independent histograms in one file pass
- grouped histograms, for example one histogram per PDG code
- computed quantities such as `pt`, `p`, `mt`, and `y_rap`
- optional skipping of elastic two-particle events
- ready-made analyses and command-line tools for running them over many SMASH
  runs (see the [user guide](docs/guide.md))

## Installation

For development, with the test dependencies:

```bash
pip install -e ".[test]"
```

The GP emulation (`brass.emulation`) needs scikit-learn, available as an extra:

```bash
pip install -e ".[emulation]"
```

Or from PyPI:

```bash
pip install pybrass
```

After rebuilding the C++ extension, restart your Python kernel.

## Examples

The `examples/` directory contains:

```text
examples/
  grouped_1d_yrap_mt.py
  grouped_2d_yrap_mt_projection.py
```

`grouped_1d_yrap_mt.py` creates grouped 1D histograms for rapidity and transverse mass in one file pass.

`grouped_2d_yrap_mt_projection.py` creates a grouped 2D histogram in `(y_rap, mT)` and projects it to `dN/dy` and `dN/dmT`.

Set `filename` at the top of each script to one of your SMASH output files.

## Analyses and command-line tools

BRASS includes ready-made analyses (`bulk`, `centrality`, `dileptons`) and
command-line tools to run them over many SMASH runs (`brass`), combine
results (`brass-merge`) and generate parameter scans (`brass-scan`):

```bash
brass bulk runs/* --pdgs 211 -211 2212 -o bulk.pkl
```

The [user guide](docs/guide.md) explains each analysis and tool, how to read
the results, and how to write your own analysis.

## Quantity order matters

The quantity list passed to `BinaryReader` must match the exact order written by SMASH.

For example:

```yaml
Output:
  Particles:
    Format: [Binary]
    Quantities: [mass, p0, pz, px, py, pdg, ncoll]
```

requires:

```python
QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]
```

Do not reorder the quantities.

## Read particles

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

while True:
    block = reader.read()

    if block is None:
        break

    if isinstance(block, brass.ParticleBlock):
        particles = block.particles

        px = particles.column("px")
        pdg = particles.column("pdg")

        print(particles.size(), pdg[:5])
```

Columns are NumPy arrays backed by C++ memory.

## Skip elastic two-particle events

```python
reader = brass.BinaryReader(
    "particles_binary.bin",
    QUANTITIES,
    skip_elastic=True,
)
```

This skips particle blocks with exactly two particles.

## Histogram rapidity

Histograms are configured with `HistogramRequest`.

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

request = brass.HistogramRequest()
request.quantities = ["y_rap"]
request.axes = [brass.RegularAxis(80, -5.0, 5.0)]

hist = brass.histogram(reader, request)

print(hist.values)
print(hist.edges)
```

## Grouped histogram

Grouped histograms use `HistogramGroupBy`.

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

request = brass.HistogramRequest()
request.quantities = ["y_rap"]
request.axes = [brass.RegularAxis(80, -5.0, 5.0)]
request.group_by = brass.HistogramGroupBy("pdg", [2212, -2212])

hists = brass.histogram(reader, request)

proton = hists[2212]
anti_proton = hists[-2212]
```

## Multiple histograms in one file pass

Use `brass.histograms(...)` to fill several independent histograms while reading the file only once.

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

y_request = brass.HistogramRequest()
y_request.quantities = ["y_rap"]
y_request.axes = [brass.RegularAxis(80, -5.0, 5.0)]

mt_request = brass.HistogramRequest()
mt_request.quantities = ["mt"]
mt_request.axes = [brass.RegularAxis(80, 0.0, 3.0)]

hist_y, hist_mt = brass.histograms(reader, [y_request, mt_request])
```

Each request creates one histogram. A request can be 1D, 2D, 3D, grouped, or ungrouped.

## Two-dimensional histogram

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

request = brass.HistogramRequest()
request.quantities = ["y_rap", "mt"]
request.axes = [
    brass.RegularAxis(100, -5.0, 5.0),
    brass.RegularAxis(100, 0.0, 3.0),
]

hist = brass.histogram(reader, request)

print(hist.shape)
print(hist.values.shape)
print(hist.edges)
```

## Grouped two-dimensional histogram

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader(
    "particles_binary.bin",
    QUANTITIES,
    skip_elastic=True,
)

request = brass.HistogramRequest()
request.quantities = ["y_rap", "mt"]
request.axes = [
    brass.RegularAxis(100, -5.0, 5.0),
    brass.RegularAxis(100, 0.0, 3.0),
]
request.group_by = brass.HistogramGroupBy(
    "pdg",
    [211, -211, 321, -321, 2212, -2212],
)

hists = brass.histogram(reader, request)

pion_plus_hist = hists[211]
proton_hist = hists[2212]
```

## Computed quantities

BRASS can compute derived quantities from the columns loaded by `BinaryReader`.

```text
pt:    px, py
p:     px, py, pz
m:     p0, px, py, pz
mt:    p0, px, py, pz
y_rap: p0, pz
```

The required input columns must be included in the `BinaryReader` quantity list.

## API summary

```python
reader = brass.BinaryReader(filename, quantities)
reader = brass.BinaryReader(filename, quantities, skip_elastic=True)

block = reader.read()

particles.column("px")
particles.columns()

request = brass.HistogramRequest()
request.quantities = ["y_rap"]
request.axes = [brass.RegularAxis(80, -5.0, 5.0)]

hist = brass.histogram(reader, request)
hist = brass.histogram(particles, request)

request.group_by = brass.HistogramGroupBy("pdg", [211, -211])

hists = brass.histogram(reader, request)
hists = brass.histogram(particles, request)

hist_y, hist_mt = brass.histograms(reader, [y_request, mt_request])
hist_y, hist_mt = brass.histograms(particles, [y_request, mt_request])
```

## Performance

Rapidity histogram (`dN/dy`, 100 bins) of every particle in a synthetic SMASH
binary file, BRASS compared with a plain NumPy reader. Both files hold the
same 2 million particles, split into small or large events:

| Case | Events × particles | File size | BRASS | NumPy | Speedup |
|---|---|---|---|---|---|
| pp-like | 200,000 × 10 | 102 MB | 0.06 s | 0.44 s | 7.0× |
| heavy-ion-like | 2,000 × 1,000 | 96 MB | 0.03 s | 0.03 s | 1.2× |

The NumPy reader loops over blocks in Python, reads each with
`np.frombuffer`, and histograms all rapidities in one `np.histogram` call at the
end. Both readers produce identical histograms.

- With many small events, NumPy spends most of its time in the per-block Python
  loop, so BRASS is several times faster.
- With large events, NumPy's vectorized math is efficient and both readers
  are limited mostly by reading the file.
- BRASS fills several histograms, or one per PDG code, in the same pass over
  the file, which the timings above do not include.

Best of 3 runs with the file in the OS cache, on an Apple M3 (16 GB),
Python 3.12, NumPy 2.5. To reproduce:

```bash
python benchmarks/benchmark.py --compare
```

`python benchmarks/benchmark.py` runs a more detailed set of BRASS timings.

## Development

```bash
pip install -e ".[test]"
pytest tests
```

After changing C++ bindings, restart Python.
