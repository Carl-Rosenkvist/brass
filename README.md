# BRASS

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

## Installation

For development:

```bash
pip install -e .
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
  1dhist.py
  2dhist.py
```

`1dhist.py` creates grouped 1D histograms for rapidity and transverse mass in one file pass.

`2dhist.py` creates a grouped 2D histogram in `(y_rap, mT)` and projects it to `dN/dy` and `dN/dmT`.

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

## Development

```bash
pip install -e .
pytest tests
```

After changing C++ bindings, restart Python.
