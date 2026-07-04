import os
import struct
import tempfile

import pytest

import brass


def write_header(f):
    f.write(b"SMSH")
    f.write(struct.pack("<H", 9))  # format_version
    f.write(struct.pack("<H", 1))  # format_variant

    version = b"SMASH-3.1"
    f.write(struct.pack("<I", len(version)))
    f.write(version)


def write_particle_block(f):
    f.write(b"p")
    f.write(struct.pack("<i", 42))  # event_number
    f.write(struct.pack("<i", 7))  # ensemble_number
    f.write(struct.pack("<I", 2))  # npart

    # particle 0: px, pdg
    f.write(struct.pack("<d", 1.25))
    f.write(struct.pack("<i", 211))

    # particle 1: px, pdg
    f.write(struct.pack("<d", -3.5))
    f.write(struct.pack("<i", -211))


def write_end_block(f):
    f.write(b"f")
    f.write(struct.pack("<I", 42))  # event_number
    f.write(struct.pack("<i", 7))  # ensemble_number
    f.write(struct.pack("<d", 1.5))  # impact_parameter
    f.write(struct.pack("<b", 1))  # empty


def make_test_file():
    tmpdir = tempfile.TemporaryDirectory()
    path = os.path.join(tmpdir.name, "test.bin")

    with open(path, "wb") as f:
        write_header(f)
        write_particle_block(f)
        write_end_block(f)

    return tmpdir, path


def test_quantity_size_helper():
    assert brass.particle_size_from_quantities(["px", "pdg"]) == 12
    assert brass.particle_size_from_quantities(["px", "py", "pz"]) == 24
    assert brass.particle_size_from_quantities([]) == 0


def test_unknown_quantity_raises():
    with pytest.raises(RuntimeError, match="unknown quantity"):
        brass.particle_size_from_quantities(["px", "does_not_exist"])


def test_binary_reader_reads_blocks():
    tmpdir, path = make_test_file()

    with tmpdir:
        reader = brass.BinaryReader(path, ["px", "pdg"])

        assert reader.header.format_version == 9
        assert reader.header.format_variant == 1
        assert reader.header.smash_version == "SMASH-3.1"

        block = reader.read()
        assert isinstance(block, brass.ParticleBlock)
        assert block.event_number == 42
        assert block.ensemble_number == 7

        assert block.particles.size() == 2
        assert block.particles.empty() is False
        assert block.particles.particle_size() == 12
        assert block.particles.particle_size() == brass.particle_size_from_quantities(
            ["px", "pdg"]
        )

        cols = block.particles.columns()

        assert set(cols) == {"px", "pdg"}
        assert cols["px"].tolist() == [1.25, -3.5]
        assert cols["pdg"].tolist() == [211, -211]

        assert block.particles.column("px").tolist() == [1.25, -3.5]
        assert block.particles.column("pdg").tolist() == [211, -211]

        block = reader.read()
        assert isinstance(block, brass.EndBlock)
        assert block.event_number == 42
        assert block.ensemble_number == 7
        assert block.impact_parameter == 1.5
        assert block.empty is True

        assert reader.read() is None
