# tests/test_xsection.py
import math
import csv
import numpy as np
import brass as br

from writing_utils import (
    writeHeader,
    writeParticleBlock,
    writeInteractionBlock,
    writeEndBlock,
)

# ---- EXACT per-particle layout required by your accessor ----
QUANTITIES = ["t", "x", "y", "z", "p0", "px", "py", "pz"]  # must match order & fields


# ---- pp cross section helpers ----
def plab_from_s(s):
    """
    Fixed-target pp: s = 2 m^2 + 2 m sqrt(p_lab^2 + m^2)
    => p_lab = sqrt(s^2 - 4 s m^2) / (2 m)
    """
    m_p = 0.938
    s = np.asarray(s, dtype=float)
    rad = s * s - 4.0 * s * (m_p**2)
    rad = np.clip(rad, 0.0, None)
    return np.sqrt(rad) / (2.0 * m_p)


def pp_total(mandelstam_s):
    p_lab = plab_from_s(mandelstam_s)
    out = np.empty_like(p_lab)

    mask1 = p_lab < 0.4
    mask2 = (p_lab >= 0.4) & (p_lab < 0.8)
    mask3 = (p_lab >= 0.8) & (p_lab < 1.5)
    mask4 = (p_lab >= 1.5) & (p_lab < 5.0)
    mask5 = p_lab >= 5.0

    out[mask1] = 34.0 * (p_lab[mask1] / 0.4) ** (-2.104)
    out[mask2] = 23.5 + 1000.0 * (p_lab[mask2] - 0.7) ** 4
    out[mask3] = 23.5 + 24.6 / (1.0 + np.exp(-(p_lab[mask3] - 1.2) / 0.1))
    out[mask4] = 41.0 + 60.0 * (p_lab[mask4] - 0.9) * np.exp(-1.2 * p_lab[mask4])
    logp = np.log(np.clip(p_lab[mask5], 1e-12, None))
    out[mask5] = 48.0 + 0.522 * logp * logp - 4.51 * logp
    return out


# ---- event generation (COM frame) ----
MP = 0.938


def two_protons_8tuple(sqrts, R):
    """
    Return two particles as 8-tuples in EXACT layout:
    (t, x, y, z, p0, px, py, pz)
    Positions are ±R/2 along x so transverse_distance = R.
    """
    E = sqrts / 2.0
    pz = math.sqrt(max(E * E - MP * MP, 0.0))
    A = (0.0, -R / 2.0, 0.0, 0.0, E, 0.0, 0.0, +pz)
    B = (0.0, +R / 2.0, 0.0, 0.0, E, 0.0, 0.0, -pz)
    return A, B


def b_sigma_from_mb(sig_mb):
    # matches your calc_tot_xs convention (R in fm; 1 fm^2 = 10 mb)
    return math.sqrt(max(sig_mb, 0.0) / (math.pi * 10.0))


# ---- the test ----
def test_xsection_matches_pp_total(tmp_path):
    rng = np.random.default_rng(1234)
    energies = [2.15,3.0, 5.0, 10.0]  # GeV
    N_per_energy = 500_000
    bmax = 2.5  # must match Xsection default

    bin_path = tmp_path / "events.bin"
    with open(bin_path, "wb") as f:
        writeHeader(f)
        ensemble = 0
        ev_id = 0

        for sqrts in energies:
            sigma_target_mb = float(pp_total(sqrts * sqrts))
            b_sig = b_sigma_from_mb(sigma_target_mb)

            for _ in range(N_per_energy):
                # R ~ uniform in area on [0, bmax]
                R = bmax * math.sqrt(rng.random())

                # particle block — SAME 8-field layout as QUANTITIES
                A, B = two_protons_8tuple(sqrts, R)
                writeParticleBlock(f, ev_id, ensemble, [A, B])

                # interaction block only when R < b_sigma — SAME 8-field layout
                if R < b_sig:
                    writeInteractionBlock(
                        f,
                        incoming=[A, B],  # EXACT layout (no PDG)
                        outgoing=[],  # not used by xsection
                        rho=0.0,
                        sigma=0.0,
                        sigma_p=0.0,
                        process=0,
                    )
                    empty = False
                else:
                    empty = True

                writeEndBlock(f, ev_id, ensemble, impact_parameter=R, empty=empty)
                ev_id += 1

    out_dir = tmp_path / "results"
    out_dir.mkdir()

    # Run your pipeline (NO CONFIG)
    br.run_analysis(
        file_and_meta=[(str(bin_path), "src=synthetic")],
        analysis_names=["xsection"],
        quantities=QUANTITIES,  # EXACT match (order + fields)
        output_folder=str(out_dir),
    )

    # Read the CSV produced by your Xsection.save()
    csv_path = out_dir / "xsection.csv"
    assert csv_path.is_file(), "xsection.csv not found"
    
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {"sqrt_s": float(r["sqrt_s"]), "xsection_mb": float(r["xsection_mb"])}
            )

    # Compare reconstructed σ to target σ for each energy
    for sqrts in energies:
        target = float(pp_total(sqrts * sqrts))
        row = min(rows, key=lambda rr: abs(rr["sqrt_s"] - sqrts))
        rec = row["xsection_mb"]
        rel = abs(rec - target) / max(target, 1e-12)
        assert (
            rel < 0.01
        ), f"√s={sqrts}: rec={rec:.2f} mb vs target={target:.2f} mb (Δ={rel:.2%})"
