# BRASS

**Binary Reader and Analysis Suite Software**

BRASS is a small C++/Python library for reading and analyzing SMASH binary particle output files.

It provides:

- a fast C++ binary reader
- zero-copy NumPy access to particle columns
- C++ histogramming
- grouped histograms, for example one histogram per PDG code
- computed quantities such as `pt`, `mt`, and `y_rap`

## Installation

```bash
pip install -e .
```

or:

```bash
pip install pybrass
```

After rebuilding the C++ extension, restart your Python kernel.

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

Columns are NumPy arrays.

## Histogram rapidity

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

hist = brass.histogram(
    reader,
    ["y_rap"],
    [brass.RegularAxis(80, -5.0, 5.0)],
)

print(hist.values)
print(hist.edges)
```

## Histogram by PDG

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

hists = brass.histograms_by(
    reader,
    ["y_rap"],
    [brass.RegularAxis(80, -5.0, 5.0)],
    by="pdg",
    group_values=[2212, -2212],
)

proton = hists[2212]
anti_proton = hists[-2212]

print(proton.values.sum())
print(anti_proton.values.sum())
```

## Plot `dN/dy`

```python
import matplotlib.pyplot as plt
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

hists = brass.histograms_by(
    reader,
    ["y_rap"],
    [brass.RegularAxis(80, -5.0, 5.0)],
    by="pdg",
    group_values=[2212, -2212],
)

n_events = reader.particle_blocks_read

proton = hists[2212]
anti_proton = hists[-2212]

edges = proton.edges[0]
centers = [(lo + hi) / 2.0 for lo, hi in zip(edges[:-1], edges[1:])]
dy = edges[1] - edges[0]

dn_dy_proton = proton.values / (dy * n_events)
dn_dy_anti_proton = anti_proton.values / (dy * n_events)

plt.step(centers, dn_dy_proton, where="mid", label=r"$p$")
plt.step(centers, dn_dy_anti_proton, where="mid", label=r"$\bar{p}$")

plt.xlabel("Rapidity y")
plt.ylabel(r"$dN/dy$")
plt.legend()
plt.tight_layout()
plt.show()
```

## Histogram a particle block

```python
import brass

QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]

reader = brass.BinaryReader("particles_binary.bin", QUANTITIES)

block = reader.read()

hist = brass.histogram(
    block.particles,
    ["y_rap"],
    [brass.RegularAxis(80, -5.0, 5.0)],
)

print(hist.values.sum())
```

For production use, prefer passing the `BinaryReader` directly to `histogram` or `histograms_by`.

## API

```python
reader = brass.BinaryReader(filename, quantities)

block = reader.read()

particles.column("px")
particles.columns()

hist = brass.histogram(reader, histogram_quantities, axes)
hist = brass.histogram(particles, histogram_quantities, axes)

hists = brass.histograms_by(
    reader,
    histogram_quantities,
    axes,
    by="pdg",
    group_values=[211, -211],
)

hists = brass.histograms_by(
    particles,
    histogram_quantities,
    axes,
    by="pdg",
    group_values=[211, -211],
)
```

## Development

```bash
pip install -e .
pytest tests
```

After changing C++ bindings, restart Python.
