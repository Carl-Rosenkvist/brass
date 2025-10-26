import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plab_from_s(s):
    """
    Fixed-target pp: s = m^2 + m^2 + 2 m E_lab = 2 m^2 + 2 m sqrt(p_lab^2 + m^2)
    => p_lab = sqrt( s^2 - 4 s m^2 ) / (2 m)
    """
    s = np.asarray(s, dtype=float)
    rad = s*s - 4.0*s*(m_p**2)
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
    out[mask2] = 23.5 + 1000.0 * (p_lab[mask2] - 0.7) ** 4
    out[mask3] = 23.5 + 24.6 / (1.0 + np.exp(-(p_lab[mask3] - 1.2) / 0.1))
    out[mask4] = 41.0 + 60.0 * (p_lab[mask4] - 0.9) * np.exp(-1.2 * p_lab[mask4])
    logp = np.log(np.clip(p_lab[mask5], 1e-12, None))
    out[mask5] = 48.0 + 0.522 * logp * logp - 4.51 * logp
    return out

