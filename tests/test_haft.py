"""The HAFT port against the Fortran HAFT 2.0, on a synthetic acceptance file."""

import struct

import numpy as np
import pytest

from brass.haft import ELECTRON, POSITRON, HadesAcceptance, kernel


def write_acc(path, seed=5, resolution=False):
    """Small HAFT file: e- and e+ matrices 6x5x4 over p 0-1.2, theta 10-80, phi 0-60."""
    rng = np.random.default_rng(seed)
    with open(path, "wb") as f:
        f.write(b"synthetic test acceptance".ljust(80))
        f.write(struct.pack("<9f", 3, 3, 3, 17, 3, 1, 0.0035, 0.0035, 0.02))
        for pid in (3, 2):
            dims = (6, 5, 4)
            f.write(struct.pack("<i", pid))
            f.write(struct.pack("<3i", *dims))
            f.write(struct.pack("<6f", 0.0, 1.2, 10.0, 80.0, 0.0, 60.0))
            f.write(rng.random(np.prod(dims)).astype("<f4").tobytes())
        if resolution:  # constant asymmetric response for e-: pos, sig, left, right, farleft
            f.write(struct.pack("<i", -3))
            f.write(struct.pack("<2i", 2, 2))
            f.write(struct.pack("<4f", 0.0, 2.0, 15.0, 95.0))
            f.write(struct.pack("<i", 5))
            for value in (-0.01, 0.02, 3.0, -17.0, 1.0):
                f.write(np.full(4, value, "<f4").tobytes())
    return path


# (pid, p, theta, phi, mode, acceptance from the Fortran HAFT 2.0 on write_acc(seed=5))
FORTRAN = [
    (3, 0.35, 33.0, 12.0, -2, 0.2912546),
    (3, 0.9, 61.5, 47.0, -2, 0.3039112),
    (2, 0.6, 45.0, 130.0, -2, 0.3048795),
    (2, 1.5, 70.0, 5.0, -2, 0.5722656),
    (3, 0.35, 33.0, 12.0, 1, 0.2126534),
    (2, 0.63, 46.5, 131.0, 0, 0.2172114),  # off the grid: nearest neighbour is ambiguous on it
    (3, 0.41, 27.3, 50.2, 2, 0.4447204),
    (3, 0.9, 61.5, 47.0, -3, 0.3410159),
    (3, 0.05, 79.0, 59.0, -2, 0.0),
    (3, 0.5, 9.0, 30.0, -2, 0.0),
    (2, 0.7, 50.0, 359.0, 4, 0.0),
]


@pytest.fixture
def acceptance_file(tmp_path):
    return write_acc(tmp_path / "test.acc")


def test_reads_the_file(acceptance_file):
    filt = HadesAcceptance(acceptance_file)
    assert filt.comment == "synthetic test acceptance"
    assert set(filt.matrices) == {ELECTRON, POSITRON}
    m = filt.matrices[ELECTRON]
    assert m.dims == (6, 5, 4)
    np.testing.assert_allclose(m.step, [0.2, 14.0, 15.0])
    assert filt.radiation_length == pytest.approx(0.02)


@pytest.mark.parametrize("pid, p, theta, phi, mode, expected", FORTRAN)
def test_acceptance_matches_fortran(acceptance_file, pid, p, theta, phi, mode, expected):
    value = HadesAcceptance(acceptance_file).acceptance(pid, p, theta, phi, mode)[0]
    assert value == pytest.approx(expected, abs=2e-6)


def test_acceptance_on_arrays(acceptance_file):
    filt = HadesAcceptance(acceptance_file)
    rows = [r for r in FORTRAN if r[0] == ELECTRON and r[4] == -2]
    p, theta, phi = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    np.testing.assert_allclose(filt.acceptance(ELECTRON, p, theta, phi), [r[5] for r in rows], atol=2e-6)


def test_kernels_are_partitions_of_unity():
    for mode, offsets, u in [
        (1, (0, 1), np.linspace(0.0, 0.99, 7)),
        *[(m, (-1, 0, 1), np.linspace(-0.49, 0.49, 7)) for m in (2, -2)],
        *[(m, (-2, -1, 0, 1, 2), np.linspace(-0.49, 0.49, 7)) for m in (3, -3, 4, -4)],
    ]:
        total = sum(kernel(u - o, mode) for o in offsets)
        np.testing.assert_allclose(total, 1.0, atol=1e-12, err_msg=f"mode {mode}")


def test_gaussian_smearing(acceptance_file):
    """Without resolution tables: Gaussian momentum resolution and energy loss."""
    filt = HadesAcceptance(acceptance_file)
    rng = np.random.default_rng(1)
    p3 = np.array([0.2, 0.1, 0.5])
    p = np.linalg.norm(p3)
    p4 = np.tile([np.sqrt(p * p + 0.000511**2), *p3], (50000, 1))
    out = filt.smear(p4, ELECTRON, resolution=3, rng=rng)
    ratio = np.linalg.norm(out[:, 1:], axis=1) / p
    sigma = 0.01 * (filt.sigma_p_a[2] + filt.sigma_p_b[2] * p)
    loss = (0.0018 + 0.006 * (1 - np.exp(-(p - 0.1) / 0.078))) / p
    assert ratio.std() == pytest.approx(sigma, rel=0.03)
    assert ratio.mean() == pytest.approx(1 - loss, abs=3 * sigma / np.sqrt(50000) + 1e-4)
    # masses are kept
    np.testing.assert_allclose(out[:, 0] ** 2 - (out[:, 1:] ** 2).sum(axis=1), 0.000511**2, atol=1e-9)


def test_resolution_tables(tmp_path):
    """With resolution tables the momentum follows the asymmetric HADES response."""
    filt = HadesAcceptance(write_acc(tmp_path / "res.acc", resolution=True))
    assert ELECTRON in filt.resolutions and POSITRON not in filt.resolutions
    rng = np.random.default_rng(2)
    p3 = np.array([0.0, 0.3, 0.4])
    p4 = np.tile([0.5, *p3], (40000, 1))
    ratio = np.linalg.norm(filt.smear(p4, ELECTRON, rng=rng)[:, 1:], axis=1) / 0.5 - 1
    # the Fortran HAFT 2.0 gives median -0.0343, 5 % -0.657, 99 % 0.074 (40000 samples)
    assert np.median(ratio) == pytest.approx(-0.0343, abs=0.003)
    assert np.percentile(ratio, 5) == pytest.approx(-0.657, abs=0.03)
    assert np.percentile(ratio, 99) == pytest.approx(0.074, abs=0.006)
