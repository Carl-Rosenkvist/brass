import os, struct, random, tempfile
import numpy as np
import pytest
from brass import BinaryReader, CollectorAccessor
from writing_utils import *
# ----------- synthetic physics -----------


def generateParticles(n):
    """Return a list of particles with consistent energy: E>|pz|."""
    parts = []
    for i in range(n):
        t = random.uniform(0.0, 100.0)
        x = random.uniform(-10.0, 10.0)
        y = random.uniform(-10.0, 10.0)
        z = random.uniform(-10.0, 10.0)

        mass = random.choice([0.938, 0.139, 1.875, 0.497])
        px = random.uniform(-5.0, 5.0)
        py = random.uniform(-5.0, 5.0)
        pz = random.uniform(-5.0, 5.0)
        p0 = (mass**2 + px**2 + py**2 + pz**2) ** 0.5  # ensure E>|pz|

        pdg = random.choice([211, -211])
        pid = i
        charge = random.choice([-1, 0, 1])

        parts.append([t, x, y, z, mass, p0, px, py, pz, pdg, pid, charge])
    return parts


# =============== test ==================


@pytest.mark.parametrize(
    "layout",
    [
        [[5, 10], [1, 4, 7]],
        [[3], [2, 2], [6, 1]],
    ],
)
def test_binary_reader_with_ensemble_numbers(layout):
    random.seed(12345)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "particles_binary.bin")
        written_particles = []  # <-- keep every particle we write, in order

        with open(path, "wb") as f:
            writeHeader(f)
            for ens_idx, event_sizes in enumerate(layout):  # ensemble_number
                for ev_idx, n in enumerate(event_sizes):  # event_number
                    parts = generateParticles(n)
                    writeParticleBlock(f, ev_idx, ens_idx, parts)
                    writeEndBlock(
                        f, ev_idx, ens_idx, impact_parameter=0.0, empty=(n == 0)
                    )
                    written_particles.extend(parts)  # record

        # read via brass
        accessor = CollectorAccessor()
        fields_d = ["t", "x", "y", "z", "mass", "p0", "px", "py", "pz"]
        fields_i = ["pdg", "id", "charge"]
        reader = BinaryReader(path, fields_d + fields_i, accessor)
        reader.read()

        # grab arrays
        arr = {k: accessor.get_double_array(k) for k in fields_d}
        arr |= {k: accessor.get_int_array(k) for k in fields_i}

        # sanity: totals match
        total = sum(sum(ev) for ev in layout)
        for k in fields_d + fields_i:
            assert len(arr[k]) == total

        # reconstruct particles from arrays in the same order
        rec_particles = [
            [
                arr["t"][i],
                arr["x"][i],
                arr["y"][i],
                arr["z"][i],
                arr["mass"][i],
                arr["p0"][i],
                arr["px"][i],
                arr["py"][i],
                arr["pz"][i],
                int(arr["pdg"][i]),
                int(arr["id"][i]),
                int(arr["charge"][i]),
            ]
            for i in range(total)
        ]

        # compare to written particles
        # floats: allclose; ints: exact
        wp = np.array(written_particles, dtype=object)  # column access
        rp = np.array(rec_particles, dtype=object)

        # float columns 0..8
        for col in range(0, 9):
            assert np.allclose(
                rp[:, col].astype(float),
                wp[:, col].astype(float),
                rtol=1e-12,
                atol=1e-12,
            ), f"float column {col} mismatch"

        # int columns 9..11
        for col in (9, 10, 11):
            assert np.array_equal(rp[:, col].astype(int), wp[:, col].astype(int)), (
                f"int column {col} mismatch"
            )

        # extra physics sanity
        e, pz = arr["p0"], arr["pz"]
        assert np.all(e > np.abs(pz))  # rapidity well-defined
        y = 0.5 * np.log((e + pz) / (e - pz))
        assert np.all(np.isfinite(y))


def test_empty_events_are_allowed():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "empty_events.bin")
        # write file: ensemble 0, event 0 empty; event 1 has 3 particles
        with open(path, "wb") as f:
            writeHeader(f)
            writeParticleBlock(f, event_number=0, ensemble_number=0, particles=[])
            writeEndBlock(f, 0, 0, impact_parameter=0.0, empty=True)

            parts = generateParticles(3)
            writeParticleBlock(f, event_number=1, ensemble_number=0, particles=parts)
            writeEndBlock(f, 1, 0, impact_parameter=0.0, empty=False)

        # read via brass
        acc = CollectorAccessor()
        fields_d = ["t", "x", "y", "z", "mass", "p0", "px", "py", "pz"]
        fields_i = ["pdg", "id", "charge"]

        reader = BinaryReader(path, fields_d + fields_i, acc)
        reader.read()

        p0 = acc.get_double_array("p0")
        pz = acc.get_double_array("pz")
        pdg = acc.get_int_array("pdg")

        # only the non-empty event contributes
        assert len(p0) == len(pz) == len(pdg) == 3
        # quick sanity: rapidity well-defined
        import numpy as np

        assert np.all(p0 > np.abs(pz))
