import os, struct, random, tempfile
import numpy as np
import pytest
from brass import BinaryReader, Accessor
from writing_utils import *


class ArrayCollector(Accessor):
    def __init__(self, fields_d, fields_i):
        super().__init__()
        self._fields_d = list(fields_d)
        self._fields_i = list(fields_i)
        self._store    = {k: [] for k in (self._fields_d + self._fields_i)}
        self._in_store = {k: [] for k in (self._fields_d + self._fields_i)}
        self._out_store= {k: [] for k in (self._fields_d + self._fields_i)}

    def on_particle_block(self, block):
        arrs = dict(self.gather_block_arrays(block))
        for k, arr in arrs.items():
            self._store[k].extend(arr.tolist())

    def on_interaction_block(self, iblock):
        arrs_in  = dict(self.gather_incoming_arrays(iblock))
        arrs_out = dict(self.gather_outgoing_arrays(iblock))
        for k, arr in arrs_in.items():
            self._in_store[k].extend(arr.tolist())
        for k, arr in arrs_out.items():
            self._out_store[k].extend(arr.tolist())

    def get_double_array(self, name):
        return np.asarray(self._store[name], dtype=float)

    def get_int_array(self, name):
        return np.asarray(self._store[name], dtype=np.int32)

    def get_incoming_array(self, name):
        dt = float if name in self._fields_d else np.int32
        return np.asarray(self._in_store[name], dtype=dt)

    def get_outgoing_array(self, name):
        dt = float if name in self._fields_d else np.int32
        return np.asarray(self._out_store[name], dtype=dt)
# ----------- synthetic physics -----------

def generateParticles(n):
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
        p0 = (mass**2 + px**2 + py**2 + pz**2) ** 0.5
        pdg = random.choice([211, -211])
        pid = i
        charge = random.choice([-1, 0, 1])
        parts.append([t, x, y, z, mass, p0, px, py, pz, pdg, pid, charge])
    return parts

# =============== tests ==================

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
        written_particles = []

        with open(path, "wb") as f:
            writeHeader(f)
            for ens_idx, event_sizes in enumerate(layout):
                for ev_idx, n in enumerate(event_sizes):
                    parts = generateParticles(n)
                    writeParticleBlock(f, ev_idx, ens_idx, parts)
                    writeEndBlock(f, ev_idx, ens_idx, impact_parameter=0.0, empty=(n == 0))
                    written_particles.extend(parts)

        fields_d = ["t", "x", "y", "z", "mass", "p0", "px", "py", "pz"]
        fields_i = ["pdg", "id", "charge"]
        acc = ArrayCollector(fields_d, fields_i)
        reader = BinaryReader(path, fields_d + fields_i, acc)
        reader.read()

        arr = {k: acc.get_double_array(k) for k in fields_d}
        arr |= {k: acc.get_int_array(k) for k in fields_i}

        total = sum(sum(ev) for ev in layout)
        for k in fields_d + fields_i:
            assert len(arr[k]) == total

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

        wp = np.array(written_particles, dtype=object)
        rp = np.array(rec_particles, dtype=object)

        for col in range(0, 9):
            assert np.allclose(
                rp[:, col].astype(float),
                wp[:, col].astype(float),
                rtol=1e-12,
                atol=1e-12,
            ), f"float column {col} mismatch"

        for col in (9, 10, 11):
            assert np.array_equal(rp[:, col].astype(int), wp[:, col].astype(int)), (
                f"int column {col} mismatch"
            )

        e, pz = arr["p0"], arr["pz"]
        assert np.all(e > np.abs(pz))
        y = 0.5 * np.log((e + pz) / (e - pz))
        assert np.all(np.isfinite(y))

def test_empty_events_are_allowed():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "empty_events.bin")
        with open(path, "wb") as f:
            writeHeader(f)
            writeParticleBlock(f, event_number=0, ensemble_number=0, particles=[])
            writeEndBlock(f, 0, 0, impact_parameter=0.0, empty=True)

            parts = generateParticles(3)
            writeParticleBlock(f, event_number=1, ensemble_number=0, particles=parts)
            writeEndBlock(f, 1, 0, impact_parameter=0.0, empty=False)

        fields_d = ["t", "x", "y", "z", "mass", "p0", "px", "py", "pz"]
        fields_i = ["pdg", "id", "charge"]
        acc = ArrayCollector(fields_d, fields_i)
        reader = BinaryReader(path, fields_d + fields_i, acc)
        reader.read()

        p0 = acc.get_double_array("p0")
        pz = acc.get_double_array("pz")
        pdg = acc.get_int_array("pdg")

        assert len(p0) == len(pz) == len(pdg) == 3
        assert np.all(p0 > np.abs(pz))

def test_read_interaction_block():
    random.seed(4242)
    name_to_idx = {
        "t":0,"x":1,"y":2,"z":3,"mass":4,"p0":5,"px":6,"py":7,"pz":8,"pdg":9,"id":10,"charge":11
    }

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "with_interaction.bin")
        with open(path, "wb") as f:
            writeHeader(f)
            writeParticleBlock(f, event_number=0, ensemble_number=0, particles=[])  # optional
            incoming = generateParticles(2)
            outgoing = generateParticles(3)
            writeInteractionBlock(f, incoming, outgoing, rho=0.25, sigma=7.5, sigma_p=0.4, process=17)
            writeEndBlock(f, 0, 0, impact_parameter=1.5, empty=False)

        fields_d = ["t","x","y","z","mass","p0","px","py","pz"]
        fields_i = ["pdg","id","charge"]
        acc = ArrayCollector(fields_d, fields_i)
        reader = BinaryReader(path, fields_d + fields_i, acc)
        reader.read()

        # verify counts
        for k in fields_d + fields_i:
            inc = acc.get_incoming_array(k)
            out = acc.get_outgoing_array(k)
            assert len(inc) == len(incoming)
            assert len(out) == len(outgoing)

        # numeric comparisons
        for k in fields_d:
            idx = name_to_idx[k]
            exp_in  = np.array([p[idx] for p in incoming], dtype=float)
            exp_out = np.array([p[idx] for p in outgoing], dtype=float)
            assert np.allclose(acc.get_incoming_array(k), exp_in, rtol=1e-12, atol=1e-12)
            assert np.allclose(acc.get_outgoing_array(k), exp_out, rtol=1e-12, atol=1e-12)

        for k in fields_i:
            idx = name_to_idx[k]
            exp_in  = np.array([int(p[idx]) for p in incoming], dtype=np.int32)
            exp_out = np.array([int(p[idx]) for p in outgoing], dtype=np.int32)
            assert np.array_equal(acc.get_incoming_array(k).astype(np.int32), exp_in)
            assert np.array_equal(acc.get_outgoing_array(k).astype(np.int32), exp_out)
