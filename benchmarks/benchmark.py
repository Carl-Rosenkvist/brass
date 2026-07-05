import os
import struct
import tempfile
import time

import numpy as np

import brass


QUANTITIES = ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"]


def write_header(f):
    f.write(b"SMSH")
    f.write(struct.pack("<H", 9))
    f.write(struct.pack("<H", 1))

    version = b"SMASH-3.1"
    f.write(struct.pack("<I", len(version)))
    f.write(version)


def write_end_block(f, event_number):
    f.write(b"f")
    f.write(struct.pack("<I", event_number))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<d", 0.0))
    f.write(struct.pack("<?", False))


def write_particle_block(f, event_number, npart, particles):
    f.write(b"p")
    f.write(struct.pack("<i", event_number))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<I", npart))

    for mass, p0, pz, px, py, pdg, ncoll in particles:
        f.write(struct.pack("<d", mass))
        f.write(struct.pack("<d", p0))
        f.write(struct.pack("<d", pz))
        f.write(struct.pack("<d", px))
        f.write(struct.pack("<d", py))
        f.write(struct.pack("<i", int(pdg)))
        f.write(struct.pack("<i", int(ncoll)))


def make_benchmark_file(path, n_events=100_000, particles_per_event=20):
    rng = np.random.default_rng(12345)

    pdgs = np.array([211, -211, 321, -321, 2212, -2212], dtype=np.int32)

    with open(path, "wb") as f:
        write_header(f)

        for event_number in range(n_events):
            if event_number % 25 == 0:
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

            particles = zip(mass, p0, pz, px, py, pdg, ncoll)

            write_particle_block(f, event_number, npart, particles)
            write_end_block(f, event_number)


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
    main()
