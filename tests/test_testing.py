import brass
import numpy as np
import pytest
from brass.testing import SmashBinaryWriter, particle_dtype

# Every quantity SMASH can write (see include/quantities.h).
SMASH_QUANTITIES = [
    "t", "x", "y", "z", "mass", "p0", "px", "py", "pz", "pdg", "id",
    "charge", "ncoll", "form_time", "xsecfac", "proc_id_origin",
    "proc_type_origin", "time_last_coll", "pdg_mother1", "pdg_mother2",
    "baryon_number", "strangeness", "tau", "eta", "eta_s", "mt", "y_rap",
    "perturbative_weight",
]  # fmt: skip


@pytest.mark.parametrize("quantity", SMASH_QUANTITIES)
def test_writer_types_match_reader(quantity):
    assert particle_dtype([quantity]).itemsize == (
        brass.particle_size_from_quantities([quantity])
    )


def test_round_trip(tmp_path):
    quantities = ["p0", "px", "pdg", "ncoll"]
    path = tmp_path / "out.bin"

    particles = {
        "p0": [1.0, 2.0],
        "px": [0.5, -0.5],
        "pdg": [211, -211],
        "ncoll": [3, 0],
    }
    incoming = [[1.5, 0.1, 2212, 1]]
    outgoing = [[0.7, 0.2, 211, 2], [0.8, -0.1, -211, 2]]

    with SmashBinaryWriter(path, quantities, smash_version="SMASH-test") as w:
        w.particles(particles, event=4, ensemble=2)
        w.interaction(
            incoming, outgoing, rho=0.25, sigma=10.5, sigma_p=3.75, process=17
        )
        w.end(event=4, ensemble=2, impact_parameter=1.5, empty=True)

    reader = brass.BinaryReader(str(path), quantities)
    assert reader.header.smash_version == "SMASH-test"

    block = reader.read()
    assert isinstance(block, brass.ParticleBlock)
    assert (block.event_number, block.ensemble_number) == (4, 2)
    for q in quantities:
        np.testing.assert_array_equal(block.particles.column(q), particles[q])

    block = reader.read()
    assert isinstance(block, brass.InteractionBlock)
    assert (block.rho, block.sigma, block.sigma_p, block.process) == (
        0.25,
        10.5,
        3.75,
        17,
    )
    for particles_out, rows in [
        (block.incoming, incoming),
        (block.outgoing, outgoing),
    ]:
        for i, q in enumerate(quantities):
            np.testing.assert_array_equal(
                particles_out.column(q), [row[i] for row in rows]
            )

    block = reader.read()
    assert isinstance(block, brass.EndBlock)
    assert (block.event_number, block.impact_parameter, block.empty) == (
        4,
        1.5,
        True,
    )

    assert reader.read() is None


def test_rejects_rows_of_wrong_length(tmp_path):
    with SmashBinaryWriter(tmp_path / "out.bin", ["px", "pdg"]) as w:
        with pytest.raises(ValueError, match="needs 2 values"):
            w.particles([[0.1, 211, 7]])


def test_rejects_missing_columns(tmp_path):
    with SmashBinaryWriter(tmp_path / "out.bin", ["px", "pdg"]) as w:
        with pytest.raises(KeyError, match="pdg"):
            w.particles({"px": [0.1]})
