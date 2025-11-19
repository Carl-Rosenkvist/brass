import numpy as np
import random
import math
import brass as br

from writing_utils import (
    writeHeader,
    writeParticleBlock,
    writeEndBlock,
)


def make_particle(pdg, ncoll, mass=0.139):
    """Generate a single particle with random momentum."""
    px = random.uniform(-0.4, 0.4)
    py = random.uniform(-0.4, 0.4)
    pz = random.uniform(-0.4, 0.4)
    p0 = math.sqrt(mass * mass + px*px + py*py + pz*pz)
    return (pdg, p0, px, py, pz, ncoll)


def write_binary(events, f):
    """Write SMASH-style particle blocks."""
    writeHeader(f)
    impact_b = 0.0
    ensemble = 0
    for i, ev in enumerate(events):
        # convert to format expected by writeParticleBlock (pdg,p0,px,py,pz,ncoll)
        writeParticleBlock(f, i, ensemble, ev)
        writeEndBlock(f, i, ensemble, impact_b, empty=False)


def test_dndydmt_unwounded(tmp_path):

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    bin_path = run_dir / "particles.bin"
    with open(bin_path, "wb") as f:

        # --------------------------------------------------------------
        # Build two artificial events
        # Event 1:
        #   - 1 unwounded proton (ncoll=0)
        #   - 1 wounded neutron (ncoll=2)
        #   - 2 pions (tracked PDGs)
        #
        # Event 2:
        #   - 2 unwounded nucleons (both ncoll=0)
        #   - 1 pion
        # --------------------------------------------------------------

        ev1 = [
            make_particle(2212, 0, mass=0.938),   # unwounded proton
            make_particle(2112, 2, mass=0.939),   # wounded neutron
            make_particle(211, 0),                # pi+
            make_particle(-211, 0),               # pi-
        ]

        ev2 = [
            make_particle(2212, 0, mass=0.938),   # unwounded p
            make_particle(2112, 0, mass=0.939),   # unwounded n
            make_particle(211, 3),                # pion
        ]

        write_binary([ev1, ev2], f)

    # --------------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------------

    state = br.run_analysis_one_file(
        filename=str(bin_path),
        meta="M=1",
        analysis_name="DndydmtUnwounded",
        quantities=["pdg", "p0", "px", "py", "pz", "ncoll"],
        opts=None,
    )

    assert "DndydmtUnwounded" in state["M=1"]
    d = state["M=1"]["DndydmtUnwounded"]

    per_unw = d["per_unwounded"]

    # --------------------------------------------------------------
    # Check unwounded counts:
    #
    # Event 1 had 1 unwounded nucleon → class 1
    # Event 2 had 2 unwounded nucleons → class 2
    # --------------------------------------------------------------

    assert 1 in per_unw
    assert 2 in per_unw

    assert per_unw[1]["n_events"] == 1
    assert per_unw[2]["n_events"] == 1

    # --------------------------------------------------------------
    # Check PDG maps exist
    # --------------------------------------------------------------

    pdg_map_1 = per_unw[1]["per_pdg"]  # class with 1 unwounded
    pdg_map_2 = per_unw[2]["per_pdg"]  # class with 2 unwounded

    # pions should appear in both
    assert 211 in pdg_map_1 or -211 in pdg_map_1
    assert 211 in pdg_map_2 or -211 in pdg_map_2

    # nucleons appear ONLY in classes where they were tracked & present
    # (proton was tracked and present in ev1 and ev2)
    assert 2212 in pdg_map_1 or 2212 in pdg_map_2

    # --------------------------------------------------------------
    # Validate histogram content for one PDG (pion in class 1)
    # --------------------------------------------------------------

    # Gather the particles manually from ev1 (unwounded count = 1)
    ev1_particles = [ev1[i] for i in range(len(ev1))]
    ev1_mt_vals = []
    ev1_y_vals = []

    for (pdg, p0, px, py, pz, ncoll) in ev1_particles:
        if pdg != 211:  # check pi+ only
            continue

        if p0 <= abs(pz):
            continue

        mt = math.sqrt(max(p0*p0 - pz*pz, 0.0))
        y = 0.5 * math.log((p0 + pz) / (p0 - pz))
        ev1_mt_vals.append(mt)
        ev1_y_vals.append(y)

    H = pdg_map_1[211]  # the analysis histogram for class1,pdg211

    # Histogram check
    mt_edges = H.edges[0]
    y_edges = H.edges[1]

    counts_truth, _, _ = np.histogram2d(
        ev1_mt_vals, ev1_y_vals,
        bins=[mt_edges, y_edges]
    )

    assert np.array_equal(H.counts, counts_truth)

    print("Test passed: DndydmtUnwounded produces correct binning and unwounded grouping.")
