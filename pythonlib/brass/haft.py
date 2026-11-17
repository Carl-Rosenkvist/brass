"""HADES acceptance filter for theorists (HAFT), single-lepton part, in NumPy.

A port of the single-particle module of HAFT 2.0 (readHAFT2.f90 by
R. Holzmann, J. Weil): reading a HADES acceptance file, the interpolated
acceptance of a particle, and the smearing of its momentum with the HADES
resolution. brass ships the acceptance files of the HADES beam times its
dilepton data comes from (haft_acceptance/, see the README there); they are
found by beam time, e.g. "p35p" for p+p at 3.5 GeV, or any .acc file by path.

    from brass.haft import HadesAcceptance

    filt = HadesAcceptance("p35p")      # or a path to a .acc file
    acc = filt.acceptance(ELECTRON, p, theta_deg, phi_deg)     # arrays
    p4 = filt.smear(p4, ELECTRON, resolution=3, rng=rng)       # (N, 4): E, px, py, pz

Frame: lab frame with z the beam axis and y the vertical axis. Angles are in
degrees; phi is folded into one HADES sector (0-60 deg). Everything works on
arrays of particles. Differences to the Fortran: double instead of single
precision (acceptances agree to ~2e-6; nearest-neighbour mode can pick the
other cell for points exactly on a cell edge), and NumPy random numbers
(smearing agrees in distribution).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSITRON, ELECTRON = 2, 3  # HAFT particle ids
ACCEPTANCE_DIR = Path(__file__).with_name("haft_acceptance")


def acceptance_files() -> dict[str, Path]:
    """Shipped acceptance files by beam time (the second part of the file name)."""
    return {path.stem.split("-")[1]: path for path in sorted(ACCEPTANCE_DIR.glob("*.acc"))}


def acceptance_path(name: str | Path) -> Path:
    """A .acc file: a path, or a shipped file by beam time ("p35p") or file name."""
    path = Path(name)
    if path.is_file():
        return path
    for beam_time, shipped in acceptance_files().items():
        if str(name) in (beam_time, shipped.name, shipped.stem):
            return shipped
    raise FileNotFoundError(
        f"no HADES acceptance file {name!r}: give a path or one of {', '.join(acceptance_files())}"
    )
LG10 = np.log(10.0)


# --- interpolation kernels -------------------------------------------------


def kernel(u: np.ndarray, mode: int) -> np.ndarray:
    """HAFT interpolation kernel (mode 0, 1, 2, -2, 3, -3, 4, -4)."""
    ua = np.abs(u)
    if mode == 0:
        return ((u > -0.5) & (u <= 0.5)).astype(float)
    if mode == 1:
        return np.where(ua < 1.0, 1.0 - ua, 0.0)
    if mode == 2:
        return np.where(ua <= 0.5, 1.0 - 2.0 * ua * ua, np.where(ua <= 1.5, 1.5 + (ua - 2.5) * ua, 0.0))
    if mode == -2:
        return np.where(ua <= 0.5, 0.75 - ua * ua, np.where(ua <= 1.5, 1.125 + 0.5 * (ua - 3.0) * ua, 0.0))
    if mode == 3:
        return np.where(ua <= 1.0, 1.0 + (ua - 2.0) * ua * ua,
                        np.where(ua <= 2.0, 4.0 + ((5.0 - ua) * ua - 8.0) * ua, 0.0))
    if mode == -3:
        return np.where(ua <= 1.0, 2.0 / 3.0 + (0.5 * ua - 1.0) * ua * ua,
                        np.where(ua <= 2.0, 4.0 / 3.0 + ((1.0 - ua / 6.0) * ua - 2.0) * ua, 0.0))
    if mode == 4:
        return np.where(ua <= 1.0, 1.0 + (1.5 * ua - 2.5) * ua * ua,
                        np.where(ua <= 2.0, 2.0 + ((2.5 - 0.5 * ua) * ua - 4.0) * ua, 0.0))
    if mode == -4:
        return np.where(ua <= 1.0, 8.0 / 9.0 + (7.0 / 6.0 * ua - 2.0) * ua * ua,
                        np.where(ua <= 2.0, 16.0 / 9.0 + ((2.0 - 7.0 / 18.0 * ua) * ua - 10.0 / 3.0) * ua, 0.0))
    raise ValueError(f"unknown interpolation mode {mode}")


def _span(mode: int) -> tuple[int, int]:
    """Summation offsets (low, high) around the floor index."""
    if mode in (0, 1):
        return 0, 1
    if mode in (2, 3, 4, -2, -3, -4):
        return -1, 2
    raise ValueError(f"unknown interpolation mode {mode}")


# --- file contents -----------------------------------------------------------


@dataclass
class _Matrix:
    dims: tuple[int, int, int]  # p, theta, phi
    lo: np.ndarray              # pmin, thmin, phmin
    hi: np.ndarray              # pmax, thmax, phmax
    values: np.ndarray          # shape (nphi, ntheta, np) like the Fortran linear order

    @property
    def step(self) -> np.ndarray:
        return (self.hi - self.lo) / np.array(self.dims)


@dataclass
class _Resolution:
    dims: tuple[int, int]       # p, theta
    lo: np.ndarray              # pmin, thmin
    hi: np.ndarray              # pmax, thmax
    tables: np.ndarray          # shape (ntab, ntheta, np)

    @property
    def step(self) -> np.ndarray:
        return (self.hi - self.lo) / np.array(self.dims)


class HadesAcceptance:
    """A HADES single-particle acceptance file (.acc).

    Attributes: comment, sigma_p_a and sigma_p_b (Gaussian momentum
    resolution per mode, in %), sigma_theta, sigma_phi, radiation_length
    (X/X0 for multiple scattering), matrices and resolutions (per HAFT id).
    """

    def __init__(self, path: str | Path):
        self.path = acceptance_path(path)
        self.matrices: dict[int, _Matrix] = {}
        self.resolutions: dict[int, _Resolution] = {}
        data = self.path.read_bytes()
        self.comment = data[:80].decode("latin-1").strip()
        head = np.frombuffer(data, "<f4", 9, 80).astype(float)
        self.sigma_p_a, self.sigma_p_b = head[0:3], head[3:6]
        self.sigma_theta, self.sigma_phi, self.radiation_length = head[6], head[7], head[8]

        pos = 116
        while pos < len(data):
            pid = int(np.frombuffer(data, "<i4", 1, pos)[0])
            pos += 4
            if pid >= 0:
                dims = tuple(int(x) for x in np.frombuffer(data, "<i4", 3, pos))
                pos += 12
                limits = np.frombuffer(data, "<f4", 6, pos).astype(float)
                pos += 24
                n = dims[0] * dims[1] * dims[2]
                values = np.frombuffer(data, "<f4", n, pos).astype(float)
                pos += 4 * n
                self.matrices[pid] = _Matrix(
                    dims, limits[0::2].copy(), limits[1::2].copy(),
                    values.reshape(dims[2], dims[1], dims[0]),
                )
            else:
                dims = tuple(int(x) for x in np.frombuffer(data, "<i4", 2, pos))
                pos += 8
                limits = np.frombuffer(data, "<f4", 4, pos).astype(float)
                pos += 16
                ntab = int(np.frombuffer(data, "<i4", 1, pos)[0])
                pos += 4
                n = dims[0] * dims[1]
                tables = np.frombuffer(data, "<f4", ntab * n, pos).astype(float)
                pos += 4 * ntab * n
                self.resolutions[-pid] = _Resolution(
                    dims, limits[0::2].copy(), limits[1::2].copy(),
                    tables.reshape(ntab, dims[1], dims[0]),
                )

    # --- acceptance ----------------------------------------------------------

    def acceptance(self, pid: int, p, theta, phi, mode: int = -2) -> np.ndarray:
        """Acceptance of particles with momentum p (GeV), theta and phi (degrees)."""
        p, theta, phi = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (p, theta, phi))
        if pid not in self.matrices:
            return np.zeros(np.broadcast(p, theta, phi).shape)
        m = self.matrices[pid]
        step = m.step
        p, theta, phi = np.broadcast_arrays(p, theta, phi)

        inside = (p >= m.lo[0]) & (theta >= m.lo[1]) & (theta <= m.hi[1]) & (phi >= m.lo[2])
        x = np.stack([
            np.minimum(p, m.hi[0] - 2.01 * step[0]),     # level off at high momentum
            theta,
            np.where(phi > 60.0, np.mod(phi, 60.0), phi),  # one HADES sector
        ])
        lo_off, hi_off = _span(mode)
        dims = np.array(m.dims)[:, None]
        # Fortran int() truncates towards zero
        floor = np.trunc(dims * ((x - 0.5 * step[:, None] - m.lo[:, None]) / (m.hi - m.lo)[:, None])).astype(int) + 1
        valid = inside & np.all(floor + lo_off >= 0, axis=0) & np.all(floor + hi_off <= dims + 1, axis=0)

        total = np.zeros(p.shape)
        offsets = range(lo_off, hi_off + 1)
        weights = []
        indices = []
        for axis in range(3):
            w, idx = [], []
            for o in offsets:
                i = floor[axis] + o
                u = (x[axis] - (i - 0.5) * step[axis] - m.lo[axis]) / step[axis]
                w.append(kernel(u, mode))
                idx.append(np.clip(i, 1, m.dims[axis]) - 1)
            weights.append(w)
            indices.append(idx)
        for wi, ii in zip(weights[0], indices[0]):
            for wj, ij in zip(weights[1], indices[1]):
                for wk, ik in zip(weights[2], indices[2]):
                    total += m.values[ik, ij, ii] * wi * wj * wk
        return np.where(valid, np.clip(total, 0.0, 1.01), 0.0)

    # --- momentum resolution ------------------------------------------------

    def _parameters(self, res: _Resolution, p: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated resolution parameters, shape (ntab, n)."""
        x = np.stack([np.clip(p, res.lo[0], res.hi[0]), np.clip(theta, res.lo[1], res.hi[1])])
        step = res.step
        dims = np.array(res.dims)[:, None]
        floor = np.trunc(dims * ((x - 0.5 * step[:, None] - res.lo[:, None]) / (res.hi - res.lo)[:, None])).astype(int) + 1
        out = np.zeros((res.tables.shape[0], x.shape[1]))
        for oi in (0, 1):
            i = floor[0] + oi
            wi = kernel((x[0] - (i - 0.5) * step[0] - res.lo[0]) / step[0], 1)
            for oj in (0, 1):
                j = floor[1] + oj
                wj = kernel((x[1] - (j - 0.5) * step[1] - res.lo[1]) / step[1], 1)
                ii = np.clip(i, 1, res.dims[0]) - 1
                jj = np.clip(j, 1, res.dims[1]) - 1
                out += res.tables[:, jj, ii] * wi * wj
        return out

    def smear(self, p4, pid: int, resolution: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
        """Smear four-momenta (rows E, px, py, pz) with the HADES resolution.

        resolution: 1 low (MDC 1+2), 2 medium (MDC 1+2+3), 3 high (MDC 1-4);
        used when the file has no resolution tables for the particle.
        """
        rng = rng or np.random.default_rng()
        p4 = np.array(p4, dtype=float, ndmin=2)
        e, px, py, pz = p4.T
        pt = np.hypot(px, py)
        ptot = np.sqrt(pt * pt + pz * pz)
        mass2 = np.maximum(e * e - ptot * ptot, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = np.arccos(np.where(ptot > 0, pz / ptot, 1.0))
            phi = np.where(pt > 0, np.arccos(np.where(pt > 0, px / pt, 1.0)), 0.0)
        phi = np.where(py < 0, 2 * np.pi - phi, phi)
        sinth = np.sin(theta)

        if pid in (POSITRON, ELECTRON):  # leptons: values from Ar+KCl embedding
            sig_theta = 0.00246 / ptot + 0.00045
            sig_phi = 0.00294 / ptot + 0.00059
        else:
            betainv = np.where(mass2 > 0, np.sqrt(1.0 + mass2 / ptot**2), 1.0)
            x0 = self.radiation_length
            sigms = 0.0136 * betainv / ptot * np.sqrt(x0) * (1 + 0.038 * np.log(x0))
            sig_theta = np.sqrt(self.sigma_theta**2 + sigms**2)
            sig_phi = np.sqrt(self.sigma_phi**2 + sigms**2)

        if pid in self.resolutions:
            params = self._parameters(self.resolutions[pid], ptot, np.degrees(theta))
            ptot = ptot * (1.0 + sample_momentum_spread(params, 2.0, rng))
        else:
            if resolution not in (1, 2, 3):
                raise ValueError("resolution must be 1, 2 or 3")
            sigp = 0.01 * ptot * (self.sigma_p_a[resolution - 1] + self.sigma_p_b[resolution - 1] * ptot)
            ptot = np.maximum(0.0, rng.normal(ptot, np.maximum(sigp, 0.0)))
            loss = np.full_like(ptot, 0.0018 if pid in (POSITRON, ELECTRON) else 0.0)
            above = ptot > 0.1
            if pid == POSITRON:
                loss += np.where(above, 0.005 * (1 - np.exp(-(ptot - 0.1) / 0.053)), 0.0)
            elif pid == ELECTRON:
                loss += np.where(above, 0.006 * (1 - np.exp(-(ptot - 0.1) / 0.078)), 0.0)
            ptot = ptot - loss  # energy loss in the target

        theta = np.abs(rng.normal(theta, sig_theta))
        theta = np.where(theta > np.pi, 2 * np.pi - theta, theta)
        phi = np.where(sinth > 0.01, rng.normal(phi, sig_phi / np.where(sinth > 0.01, sinth, 1.0)), phi)
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)
        phi = np.where(phi > 2 * np.pi, phi - 2 * np.pi, phi)

        st = np.sin(theta)
        out = np.stack([np.zeros_like(ptot), ptot * st * np.cos(phi), ptot * st * np.sin(phi), ptot * np.cos(theta)], axis=1)
        out[:, 0] = np.sqrt((out[:, 1:] ** 2).sum(axis=1) + mass2)
        return out


# --- asymmetric momentum response (files with resolution tables) ------------


def momentum_spread(x, par, ns):
    """HADES momentum response (p_rec - p_sim) / p_sim; par rows: pos, sig, left, right, farleft."""
    pos, sig, left, right, farleft = par[:5]
    amp = np.exp(-0.5 * ns * ns)
    edge = pos - ns * sig
    far = -LG10 / left + edge
    argn = np.where((x >= edge) | (x <= far), 0.0, 1.0)
    argp = np.where(x >= edge, 1.0, 0.0)
    argn2 = np.where(x > far, 0.0, 1.0)
    return (np.exp(-0.5 * ((x - pos) / sig) ** 2)
            + amp * np.exp(left * (x - edge)) * argn
            + amp * np.exp(right * (x - edge)) * argp
            + 0.1 * amp * np.exp(farleft * (x - far)) * argn2)


def sample_momentum_spread(par: np.ndarray, ns: float, rng: np.random.Generator) -> np.ndarray:
    """Sample (p_rec - p_sim) / p_sim per particle (par: shape (ntab, n)), by rejection as in HAFT."""
    pos, sig, left, right, farleft = par[:5]
    n = par.shape[1]
    e2 = np.exp(-0.5 * ns * ns)
    amp = np.stack([
        0.1 * (1 + e2) * np.ones(n),
        (1 + e2) * np.ones(n),
        1 + np.exp(right * ns * sig),
        (e2 + np.exp(right * 2 * ns * sig)) / np.exp(right * 2 * ns * sig),
    ])
    area = np.stack([
        amp[0] / farleft * (1 - np.exp(farleft * (LG10 / left + ns * sig - pos - 1))),
        amp[1] / left * 0.9,
        amp[2] * 2 * ns * sig,
        amp[3] / right * (np.exp(right * (1 - pos + ns * sig)) - np.exp(right * 2 * ns * sig)),
    ])
    cum = np.cumsum(area / area.sum(axis=0), axis=0)

    result = np.zeros(n)
    todo = np.arange(n)
    for _ in range(1000):
        if todo.size == 0:
            break
        r1 = rng.random(todo.size)
        region = (r1[None, :] >= cum[:, todo]).sum(axis=0)
        dx = np.zeros(todo.size)
        ftest = np.zeros(todo.size)
        po, si, le, ri, fa = (x[todo] for x in (pos, sig, left, right, farleft))
        for reg in range(4):
            sel = region == reg
            if not sel.any():
                continue
            k = np.flatnonzero(sel)
            if reg == 0:   # far left tail, limited to dx >= -1
                d = _sample_tail(rng, k.size, lambda r: r / fa[k] - ns * si[k] + po[k] - LG10 / le[k], lambda d: d >= -1.0)
                ftest[k] = amp[0, todo[k]] * np.exp(fa[k] * (d + LG10 / le[k] - po[k] + ns * si[k]))
            elif reg == 1:  # left tail
                d = _sample_tail(rng, k.size, lambda r: r / le[k] - ns * si[k] + po[k],
                                 lambda d: d >= -LG10 / le[k] - ns * si[k] + po[k])
                ftest[k] = amp[1, todo[k]] * np.exp(le[k] * (d - po[k] + ns * si[k]))
            elif reg == 2:  # peak
                d = 2 * ns * si[k] * (rng.random(k.size) - 0.5) + po[k]
                ftest[k] = amp[2, todo[k]]
            else:           # right tail, limited to dx <= 1
                d = _sample_tail(rng, k.size, lambda r: r / ri[k] + ns * si[k] + po[k], lambda d: d <= 1.0)
                ftest[k] = amp[3, todo[k]] * np.exp(ri[k] * (d - po[k] + ns * si[k]))
            dx[k] = d
        accept = rng.random(todo.size) < momentum_spread(dx, [po, si, le, ri, fa], ns) / ftest
        result[todo[accept]] = dx[accept]
        todo = todo[~accept]
    return result  # particles still undecided after 1000 trials keep 0, as in HAFT


def _sample_tail(rng, n, transform, ok):
    """Exponential tail samples with up to 1000 tries per particle (HAFT's inner loops)."""
    d = transform(np.log(rng.random(n)))
    for _ in range(999):
        bad = ~ok(d)
        if not bad.any():
            break
        d = np.where(bad, transform(np.log(rng.random(n))), d)
    return d
