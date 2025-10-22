import numpy as np
import pytest
from brass import HistND


def test_fill_1d_basic():
    edges = np.linspace(0, 1, 6)
    H = HistND([edges])
    x = np.linspace(0.01, 0.99, 10)
    H.fill(x)
    assert H.counts.sum() == 10
    assert np.allclose(H.counts, 2)


def test_fill_1d_weighted_and_scalar_weight():
    edges = np.linspace(0, 1, 6)
    H = HistND([edges])
    x = np.linspace(0.01, 0.99, 10)
    H.fill(x, weights=0.5)  # scalar weight
    assert np.isclose(H.counts.sum(), 5.0)
    H2 = HistND([edges])
    w = np.full_like(x, 0.5)
    H2.fill(x, weights=w)
    assert np.allclose(H.counts, H2.counts)


def test_variance_tracking():
    edges = np.linspace(0, 1, 6)
    H = HistND([edges], track_variance=True)
    x = np.linspace(0.01, 0.99, 10)
    w = np.full_like(x, 2.0)
    H.fill(x, weights=w)
    assert np.isclose(H.sumw2.sum(), 40.0)
    assert np.allclose(H.errors() ** 2, H.sumw2)


def test_out_of_range_ignored():
    edges = np.linspace(0, 1, 6)
    H = HistND([edges])
    x = np.array([-1, 2, 0.5])
    H.fill(x)
    assert np.isclose(H.counts.sum(), 1.0)


def test_fill_2d_uniform():
    x_edges = np.linspace(0, 1, 6)
    y_edges = np.linspace(0, 1, 4)
    H = HistND([x_edges, y_edges])
    rng = np.random.default_rng(42)
    x = rng.random(1000)
    y = rng.random(1000)
    H.fill(x, y)
    assert np.isclose(H.counts.sum(), 1000)


def test_project_2d_to_1d():
    x_edges = np.linspace(0, 1, 6)
    y_edges = np.linspace(0, 1, 4)
    H = HistND([x_edges, y_edges])
    rng = np.random.default_rng(1)
    x = rng.random(1000)
    y = rng.random(1000)
    H.fill(x, y)
    proj, edges = H.project([0])
    assert proj.shape == (len(x_edges) - 1,)
    assert np.isclose(proj.sum(), H.counts.sum())


def test_normalized_copy():
    edges = np.linspace(0, 1, 6)
    H = HistND([edges])
    x = np.linspace(0.01, 0.99, 10)
    H.fill(x)
    Hn = H.normalized_copy(10.0)
    assert np.isclose(Hn.counts.sum(), 1.0)


def test_density_1d():
    edges = np.linspace(0, 1, 6)  # width 0.2
    H = HistND([edges])
    x = np.linspace(0.01, 0.99, 10)
    H.fill(x)
    d = H.density()
    assert np.allclose(d, 10.0)  # 2 / 0.2


def test_merge_histograms_and_add():
    edges = np.linspace(0, 1, 6)
    H1 = HistND([edges])
    H2 = HistND([edges])
    x1 = np.linspace(0.01, 0.99, 10)
    x2 = np.linspace(0.01, 0.99, 5)
    H1.fill(x1)
    H2.fill(x2)
    # merge_ (+=)
    H1.merge_(H2)
    assert np.isclose(H1.counts.sum(), 15.0)
    # + returns new
    H3 = HistND([edges])
    H3.fill([0.5, 0.5])
    Hsum = H1 + H3
    assert np.isclose(Hsum.counts.sum(), 17.0)
    # radd via sum()
    total = sum([H2, H3], start=0)
    assert isinstance(total, HistND)
    assert np.isclose(total.counts.sum(), H2.counts.sum() + H3.counts.sum())


def test_add_promotes_variance_and_dtype():
    edges = np.linspace(0, 1, 3)
    Hn = HistND([edges], dtype=np.int32, track_variance=False)
    Hv = HistND([edges], dtype=np.float64, track_variance=True)
    Hn.fill([0.25, 0.75], weights=2)  # counts int32
    Hv.fill([0.25, 0.75], weights=1.5)  # counts float, sumw2
    Hs = Hn + Hv
    # dtype promoted to float64
    assert Hs.counts.dtype == np.float64
    # variance present (at least zeros where missing)
    assert hasattr(Hs, "sumw2")


def test_scalar_fill_and_mask():
    edges = np.linspace(0, 1, 6)
    H = HistND([edges])
    data = np.linspace(0, 1, 20, endpoint=False)
    mask = np.zeros_like(data, dtype=bool)
    mask[::2] = True  # keep even indices
    H.fill(data, mask=mask, weights=2.0)
    # 10 values kept, each weight 2
    assert np.isclose(H.counts.sum(), 20.0)
