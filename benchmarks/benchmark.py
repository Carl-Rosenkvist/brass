"""BRASS benchmarks on synthetic SMASH binary files.

    python benchmarks/benchmark.py            # detailed BRASS timings
    python benchmarks/benchmark.py --compare  # BRASS vs a NumPy reader
"""

import argparse
import os
import platform
import struct
import tempfile
import time

import numpy as np

import brass
from brass.testing import SmashBinaryWriter, particle_dtype


QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]


def make_benchmark_file(
    path,
    n_events=100_000,
    particles_per_event=20,
    elastic_every=25,
):
    """Write a synthetic file. Every `elastic_every`-th event has 2 particles
    (like an elastic collision); pass None for uniform events."""
    rng = np.random.default_rng(12345)

    pdgs = np.array([211, -211, 321, -321, 2212, -2212], dtype=np.int32)

    with SmashBinaryWriter(path, QUANTITIES) as writer:
        for event_number in range(n_events):
            if elastic_every and event_number % elastic_every == 0:
                npart = 2
            else:
                npart = particles_per_event

            mass = rng.uniform(0.1, 1.0, size=npart)
            px = rng.normal(0.0, 0.6, size=npart)
            py = rng.normal(0.0, 0.6, size=npart)
            pz = rng.normal(0.0, 1.0, size=npart)

            p_abs2 = px * px + py * py + pz * pz
            p0 = np.sqrt(mass * mass + p_abs2)

            pdg = rng.choice(pdgs, size=npart)
            ncoll = rng.integers(0, 20, size=npart, dtype=np.int32)

            writer.particles(
                {
                    "mass": mass,
                    "p0": p0,
                    "pz": pz,
                    "px": px,
                    "py": py,
                    "pdg": pdg,
                    "ncoll": ncoll,
                },
                event=event_number,
            )
            writer.end(event=event_number)


def bench(name, func, repeats=5):
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)

    times = np.asarray(times)

    print(f"{name}")
    print(f"  best: {times.min():.6f} s")
    print(f"  mean: {times.mean():.6f} s")
    print(f"  std:  {times.std():.6f} s")
    print()

    return result


def make_y_request():
    request = brass.HistogramRequest()
    request.quantities = ["y_rap"]
    request.axes = [brass.RegularAxis(100, -5.0, 5.0)]
    return request


def make_mt_request():
    request = brass.HistogramRequest()
    request.quantities = ["mt"]
    request.axes = [brass.RegularAxis(100, 0.0, 3.0)]
    return request


def make_y_mt_2d_request():
    request = brass.HistogramRequest()
    request.quantities = ["y_rap", "mt"]
    request.axes = [
        brass.RegularAxis(100, -5.0, 5.0),
        brass.RegularAxis(100, 0.0, 3.0),
    ]
    return request


def make_grouped_y_request():
    request = make_y_request()
    request.group_by = brass.HistogramGroupBy(
        "pdg",
        [211, -211, 321, -321, 2212, -2212],
    )
    return request


def make_grouped_mt_request():
    request = make_mt_request()
    request.group_by = brass.HistogramGroupBy(
        "pdg",
        [211, -211, 321, -321, 2212, -2212],
    )
    return request


def bench_read_all_blocks(path, skip_elastic=False):
    def run():
        reader = brass.BinaryReader(path, QUANTITIES, skip_elastic=skip_elastic)

        n_particle_blocks = 0
        n_particles = 0

        while True:
            block = reader.read()

            if block is None:
                break

            if isinstance(block, brass.ParticleBlock):
                n_particle_blocks += 1
                n_particles += block.particles.size()

        return n_particle_blocks, n_particles

    return run


def bench_column_access(path):
    def run():
        reader = brass.BinaryReader(path, QUANTITIES)

        total = 0.0
        n_particles = 0

        while True:
            block = reader.read()

            if block is None:
                break

            if not isinstance(block, brass.ParticleBlock):
                continue

            px = block.particles.column("px")
            py = block.particles.column("py")

            total += np.sum(px)
            total += np.sum(py)
            n_particles += block.particles.size()

        return total, n_particles

    return run


def bench_columns_access(path):
    def run():
        reader = brass.BinaryReader(path, QUANTITIES)

        total = 0.0
        n_particles = 0

        while True:
            block = reader.read()

            if block is None:
                break

            if not isinstance(block, brass.ParticleBlock):
                continue

            cols = block.particles.columns()

            total += np.sum(cols["px"])
            total += np.sum(cols["py"])
            n_particles += block.particles.size()

        return total, n_particles

    return run


def bench_histogram_y(path):
    def run():
        reader = brass.BinaryReader(path, QUANTITIES)
        return brass.histogram(reader, make_y_request())

    return run


def bench_histogram_mt(path):
    def run():
        reader = brass.BinaryReader(path, QUANTITIES)
        return brass.histogram(reader, make_mt_request())

    return run


def bench_histogram_y_mt_2d(path):
    def run():
        reader = brass.BinaryReader(path, QUANTITIES)
        return brass.histogram(reader, make_y_mt_2d_request())

    return run


def bench_histograms_y_and_mt_separate(path, skip_elastic=False):
    def run():
        reader = brass.BinaryReader(path, QUANTITIES, skip_elastic=skip_elastic)
        return brass.histograms(
            reader,
            [
                make_y_request(),
                make_mt_request(),
            ],
        )

    return run


def bench_grouped_histograms_y_and_mt_separate(path):
    def run():
        reader = brass.BinaryReader(path, QUANTITIES)
        return brass.histograms(
            reader,
            [
                make_grouped_y_request(),
                make_grouped_mt_request(),
            ],
        )

    return run


# ---------------------------------------------------------------------
# Comparison with a plain NumPy reader
# ---------------------------------------------------------------------

DNDY_BINS = 100
DNDY_RANGE = (-5.0, 5.0)

# Same total number of particles, very different event sizes.
COMPARE_CASES = [
    ("pp-like", 200_000, 10),
    ("heavy-ion-like", 2_000, 1_000),
]


def brass_dndy(path):
    reader = brass.BinaryReader(path, QUANTITIES)

    request = brass.HistogramRequest()
    request.quantities = ["y_rap"]
    request.axes = [brass.RegularAxis(DNDY_BINS, *DNDY_RANGE)]

    return np.asarray(brass.histogram(reader, request).values)


def numpy_dndy(path):
    """Rapidity histogram with NumPy only: read each block with
    np.frombuffer, collect rapidities, histogram once at the end."""
    dtype = particle_dtype(QUANTITIES)
    rapidities = []

    with open(path, "rb") as f:
        f.read(4)  # magic number
        _, _, version_length = struct.unpack("<HHI", f.read(8))
        f.read(version_length)

        while tag := f.read(1):
            if tag == b"p":
                _, _, npart = struct.unpack("<iiI", f.read(12))
                particles = np.frombuffer(
                    f.read(npart * dtype.itemsize), dtype=dtype
                )
                p0 = particles["p0"]
                pz = particles["pz"]
                rapidities.append(0.5 * np.log((p0 + pz) / (p0 - pz)))

            elif tag == b"f":
                f.read(4 + 4 + 8 + 1)

            else:
                raise RuntimeError(f"unexpected block tag: {tag!r}")

    return np.histogram(
        np.concatenate(rapidities), bins=DNDY_BINS, range=DNDY_RANGE
    )[0]


def best_time(func, repeats):
    best = float("inf")
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        best = min(best, time.perf_counter() - start)

    return best, result


def compare(repeats=3):
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, n_events, per_event in COMPARE_CASES:
            path = os.path.join(tmpdir, f"{name}.bin")
            make_benchmark_file(path, n_events, per_event, elastic_every=None)
            size_mb = os.path.getsize(path) / 1e6

            t_brass, h_brass = best_time(lambda: brass_dndy(path), repeats)
            t_numpy, h_numpy = best_time(lambda: numpy_dndy(path), repeats)

            if not np.array_equal(h_brass, h_numpy):
                raise RuntimeError(f"{name}: BRASS and NumPy histograms differ")

            rows.append(
                (name, n_events, per_event, size_mb, t_brass, t_numpy)
            )

    print(f"dN/dy of all particles, {DNDY_BINS} bins; best of {repeats} runs")
    print(f"{platform.machine()}, Python {platform.python_version()}, "
          f"NumPy {np.__version__}")
    print("Histograms from both readers are identical.")
    print()
    print("| Case | Events × particles | File size | BRASS | NumPy | Speedup |")
    print("|---|---|---|---|---|---|")

    for name, n_events, per_event, size_mb, t_brass, t_numpy in rows:
        print(
            f"| {name} | {n_events:,} × {per_event:,} | {size_mb:.0f} MB "
            f"| {t_brass:.2f} s | {t_numpy:.2f} s "
            f"| {t_numpy / t_brass:.1f}× |"
        )


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bench_particles.bin")

        n_events = 100_000
        particles_per_event = 20

        print("Creating benchmark file...")
        make_benchmark_file(
            path,
            n_events=n_events,
            particles_per_event=particles_per_event,
        )

        size_mb = os.path.getsize(path) / 1024**2

        print("BRASS benchmark")
        print(f"file: {path}")
        print(f"size: {size_mb:.2f} MiB")
        print(f"events: {n_events}")
        print(f"particles/event: mostly {particles_per_event}, every 25th event has 2")
        print()

        blocks, particles = bench(
            "reader.read all blocks",
            bench_read_all_blocks(path),
        )
        print(f"  particle blocks: {blocks}")
        print(f"  particles:       {particles}")
        print()

        blocks_skip, particles_skip = bench(
            "reader.read all blocks skip_elastic",
            bench_read_all_blocks(path, skip_elastic=True),
        )
        print(f"  particle blocks: {blocks_skip}")
        print(f"  particles:       {particles_skip}")
        print()

        bench("particles.column px/py", bench_column_access(path))
        bench("particles.columns all", bench_columns_access(path))

        bench("histogram y_rap", bench_histogram_y(path))
        bench("histogram mt", bench_histogram_mt(path))
        bench("histogram 2D y_rap,mt", bench_histogram_y_mt_2d(path))

        bench(
            "histograms separate y_rap + mt one pass",
            bench_histograms_y_and_mt_separate(path),
        )

        bench(
            "histograms grouped y_rap + mt by pdg one pass",
            bench_grouped_histograms_y_and_mt_separate(path),
        )

        bench(
            "histograms separate y_rap + mt one pass skip_elastic",
            bench_histograms_y_and_mt_separate(path, skip_elastic=True),
        )

    print("Temporary benchmark file removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare BRASS with a plain NumPy reader (README table)",
    )
    args = parser.parse_args()

    if args.compare:
        compare()
    else:
        main()
