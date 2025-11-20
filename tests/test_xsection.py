# tests/test_xsection.py

import math
import numpy as np
import brass as br

from writing_utils import (
    writeHeader,
    writeParticleBlock,
    writeInteractionBlock,
    writeEndBlock,
)

QUANTITIES = ["t", "x", "y", "z", "p0", "px", "py", "pz"]


# --------- pp total cross section parametrization ----------
def plab_from_s(s):
    m_p = 0.938
    s = np.asarray(s, dtype=float)
    rad = s * s - 4.0 * s * (m_p*m_p)
    rad = np.clip(rad, 0.0, None)
    return np.sqrt(rad) / (2.0*m_p)


def pp_total(mandelstam_s):
    p_lab = plab_from_s(mandelstam_s)
    out = np.empty_like(p_lab)

    mask1 = p_lab < 0.4
    mask2 = (p_lab >= 0.4) & (p_lab < 0.8)
    mask3 = (p_lab >= 0.8) & (p_lab < 1.5)
    mask4 = (p_lab >= 1.5) & (p_lab < 5.0)
    mask5 = p_lab >= 5.0

    out[mask1] = 34.0 * (p_lab[mask1] / 0.4) ** (-2.104)
    out[mask2] = 23.5 + 1000.0*(p_lab[mask2] - 0.7)**4
    out[mask3] = 23.5 + 24.6 / (1.0 + np.exp(-(p_lab[mask3] - 1.2)/0.1))
    out[mask4] = 41.0 + 60.0*(p_lab[mask4] - 0.9)*np.exp(-1.2*p_lab[mask4])
    logp = np.log(np.clip(p_lab[mask5], 1e-12, None))
    out[mask5] = 48.0 + 0.522*logp*logp - 4.51*logp
    return out


# --------- proton kinematics ----------
MP = 0.938


def two_protons_8tuple(sqrts, R):
    E = sqrts * 0.5
    pz = math.sqrt(max(E*E - MP*MP, 0.0))
    return (
        (0.0, -R/2, 0.0, 0.0, E, 0.0, 0.0, +pz),
        (0.0, +R/2, 0.0, 0.0, E, 0.0, 0.0, -pz),
    )


def b_sigma_from_mb(sig_mb):
    return math.sqrt(max(sig_mb, 0.0) / (math.pi * 10.0))


# -------------------------------------------------------------
#                     THE REAL TEST
# -------------------------------------------------------------
def test_xsection_matches_pp_total(tmp_path):
    rng = np.random.default_rng(42)

    energies = [2.15, 3.0, 5.0, 10.0]
    N_per_energy = 800000
    bmax = 2.5

    bin_path = tmp_path / "events.bin"
    with open(bin_path, "wb") as f:
        writeHeader(f)
        ev_id = 0
        ensemble = 0

        for sqrts in energies:
            sigma_target_mb = float(pp_total(sqrts*sqrts))
            b_sig = b_sigma_from_mb(sigma_target_mb)

            for _ in range(N_per_energy):
                R = bmax * math.sqrt(rng.random())
                A, B = two_protons_8tuple(sqrts, R)
                writeParticleBlock(f, ev_id, ensemble, [A, B])

                if R < b_sig:
                    writeInteractionBlock(
                        f,
                        incoming=[A, B],
                        outgoing=[],
                        rho=0.0, sigma=0.0, sigma_p=0.0,
                        process=0,
                    )
                    empty = False
                else:
                    empty = True

                writeEndBlock(f, ev_id, ensemble, impact_parameter=R, empty=empty)
                ev_id += 1

    # -------------------------------------------------------------
    # 1. RUN ANALYSIS (raw results)
    # -------------------------------------------------------------
    state = br.run_analysis_one_file(
        filename=str(bin_path),
        meta="test_xs",
        analysis_name="xsection",
        quantities=QUANTITIES,
        opts=None,
    )

    assert "test_xs" in state
    assert "xsection" in state["test_xs"]

    # -------------------------------------------------------------
    # 2. MANUALLY RUN FINALIZE (because run_analysis_one_file does NOT)
    # -------------------------------------------------------------
    # Create one dummy Xsection instance only to call finalize()
    xs = br.create_analysis("xsection")
    xs.finalize(state)

    assert "xsection_rows" in state["test_xs"], "finalized rows missing"

    rows = state["test_xs"]["xsection_rows"]
    assert isinstance(rows, list)
    assert len(rows) == len(energies)

    # -------------------------------------------------------------
    # 3. Validate results
    # -------------------------------------------------------------
    for sqrts in energies:
        row = min(rows, key=lambda r: abs(r["sqrt_s"] - sqrts))
        rec = row["xsection_mb"]
        target = float(pp_total(sqrts*sqrts))
        rel = abs(rec - target) / max(target, 1e-12)

        assert rel < 0.03, (
            f"√s={sqrts:.2f}: rec={rec:.2f} mb vs target={target:.2f} mb "
            f"(∆={rel:.2%})"
        )
