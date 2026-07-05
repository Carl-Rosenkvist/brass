import struct

import numpy as np

import brass


QUANTITIES = ["px", "py", "pz", "pdg"]
FOUR_MOMENTUM_QUANTITIES = ["p0", "px", "py", "pz"]


def histogram_request(quantities, axes, group_by=None):
    request = brass.HistogramRequest()
    request.quantities = quantities
    request.axes = axes
    request.group_by = group_by
    return request


def histogram_group_by(quantity, values):
    return brass.HistogramGroupBy(quantity, values)


def write_header(f):
    f.write(b"SMSH")
    f.write(struct.pack("<H", 9))
    f.write(struct.pack("<H", 1))

    version = b"SMASH-3.1"
    f.write(struct.pack("<I", len(version)))
    f.write(version)


def write_end_block(f):
    f.write(b"f")
    f.write(struct.pack("<I", 1))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<d", 0.0))
    f.write(struct.pack("<?", False))


def write_particle_block(f, particles):
    f.write(b"p")
    f.write(struct.pack("<i", 1))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<I", len(particles)))

    # quantities = ["px", "py", "pz", "pdg"]
    for px, py, pz, pdg in particles:
        f.write(struct.pack("<d", px))
        f.write(struct.pack("<d", py))
        f.write(struct.pack("<d", pz))
        f.write(struct.pack("<i", pdg))


def write_four_momentum_block(f, particles):
    f.write(b"p")
    f.write(struct.pack("<i", 1))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<I", len(particles)))

    # quantities = ["p0", "px", "py", "pz"]
    for p0, px, py, pz in particles:
        f.write(struct.pack("<d", p0))
        f.write(struct.pack("<d", px))
        f.write(struct.pack("<d", py))
        f.write(struct.pack("<d", pz))


def make_file(path, particles):
    with open(path, "wb") as f:
        write_header(f)
        write_particle_block(f, particles)
        write_end_block(f)


def make_four_momentum_file(path, particles):
    with open(path, "wb") as f:
        write_header(f)
        write_four_momentum_block(f, particles)
        write_end_block(f)


def make_reader(path, quantities=QUANTITIES):
    return brass.BinaryReader(str(path), quantities)


def first_particle_block(path, quantities=QUANTITIES):
    reader = make_reader(path, quantities)
    block = reader.read()
    assert isinstance(block, brass.ParticleBlock)
    return block


def assert_edges_equal(h, expected):
    assert h.edges == expected


def test_histogram_1d_px_from_reader(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, 211),
            (0.7, 0.3, 0.3, -211),
            (0.9, 0.9, 0.4, -211),
            (1.2, 0.5, 0.5, 321),
        ],
    )

    request = histogram_request(
        ["px"],
        [brass.RegularAxis(2, 0.0, 1.0)],
    )

    h = brass.histogram(make_reader(path), request)

    assert h.shape == [2]
    assert h.values.shape == (2,)
    assert h.values.sum() == 4.0
    assert np.array_equal(h.values, np.array([2.0, 2.0]))
    assert_edges_equal(h, [[0.0, 0.5, 1.0]])


def test_histogram_2d_px_py_from_reader(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, 211),
            (0.7, 0.3, 0.3, -211),
            (0.9, 0.9, 0.4, -211),
            (1.2, 0.5, 0.5, 321),
        ],
    )

    request = histogram_request(
        ["px", "py"],
        [
            brass.RegularAxis(2, 0.0, 1.0),
            brass.RegularAxis(2, 0.0, 1.0),
        ],
    )

    h = brass.histogram(make_reader(path), request)

    expected = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    assert h.shape == [2, 2]
    assert h.values.shape == (2, 2)
    assert h.values.sum() == 4.0
    assert np.array_equal(h.values, expected)

    assert_edges_equal(
        h,
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
    )


def test_histogram_3d_px_py_pz_from_reader(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, 211),
            (0.7, 0.3, 0.8, -211),
            (0.9, 0.9, 0.9, -211),
            (0.4, 0.4, 1.2, 321),
        ],
    )

    request = histogram_request(
        ["px", "py", "pz"],
        [
            brass.RegularAxis(2, 0.0, 1.0),
            brass.RegularAxis(2, 0.0, 1.0),
            brass.RegularAxis(2, 0.0, 1.0),
        ],
    )

    h = brass.histogram(make_reader(path), request)

    expected = np.zeros((2, 2, 2))
    expected[0, 0, 0] = 1.0
    expected[0, 1, 0] = 1.0
    expected[1, 0, 1] = 1.0
    expected[1, 1, 1] = 1.0

    assert h.shape == [2, 2, 2]
    assert h.values.shape == (2, 2, 2)
    assert h.values.sum() == 4.0
    assert np.array_equal(h.values, expected)

    assert_edges_equal(
        h,
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
    )


def test_histogram_from_particles_object(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, 211),
            (0.7, 0.3, 0.3, -211),
            (0.9, 0.9, 0.4, -211),
            (1.2, 0.5, 0.5, 321),
        ],
    )

    block = first_particle_block(path)

    request = histogram_request(
        ["px", "py"],
        [
            brass.RegularAxis(2, 0.0, 1.0),
            brass.RegularAxis(2, 0.0, 1.0),
        ],
    )

    h = brass.histogram(block.particles, request)

    expected = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    assert h.shape == [2, 2]
    assert h.values.shape == (2, 2)
    assert h.values.sum() == 4.0
    assert np.array_equal(h.values, expected)

    assert_edges_equal(
        h,
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
    )


def test_histogram_computed_pt_from_reader(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.3, 0.4, 0.0, 211),
            (0.6, 0.8, 0.0, -211),
            (0.1, 0.1, 0.0, 321),
        ],
    )

    request = histogram_request(
        ["pt"],
        [brass.RegularAxis(2, 0.0, 1.0)],
    )

    h = brass.histogram(make_reader(path), request)

    assert h.shape == [2]
    assert h.values.shape == (2,)
    assert h.values.sum() == 2.0
    assert np.array_equal(h.values, np.array([1.0, 1.0]))


def test_histogram_computed_mt_y_rap_from_reader(tmp_path):
    path = tmp_path / "particles_four_momentum.bin"

    particles = np.array(
        [
            (1.9, 0.3, 0.4, 0.0),
            (3.0, 0.6, 0.8, 1.0),
            (3.0, 0.1, 0.1, -1.0),
            (1.0, 0.1, 0.1, 1.0),
        ],
        dtype=float,
    )

    make_four_momentum_file(path, particles)

    mt_edges = np.linspace(0.0, 3.0, 4)
    y_edges = np.linspace(-1.0, 1.0, 3)

    request = histogram_request(
        ["mt", "y_rap"],
        [
            brass.RegularAxis(3, 0.0, 3.0),
            brass.RegularAxis(2, -1.0, 1.0),
        ],
    )

    h = brass.histogram(make_reader(path, FOUR_MOMENTUM_QUANTITIES), request)

    p0 = particles[:, 0]
    px = particles[:, 1]
    py = particles[:, 2]
    pz = particles[:, 3]

    valid = p0 > np.abs(pz)

    p0 = p0[valid]
    px = px[valid]
    py = py[valid]
    pz = pz[valid]

    pt = np.hypot(px, py)
    m2 = np.maximum(p0 * p0 - px * px - py * py - pz * pz, 0.0)
    mt = np.hypot(pt, np.sqrt(m2))
    y_rap = 0.5 * np.log((p0 + pz) / (p0 - pz))

    expected, _, _ = np.histogram2d(
        mt,
        y_rap,
        bins=[mt_edges, y_edges],
    )

    assert h.shape == [3, 2]
    assert h.values.shape == expected.shape
    assert h.values.sum() == expected.sum()
    assert np.array_equal(h.values, expected)

    assert_edges_equal(
        h,
        [
            mt_edges.tolist(),
            y_edges.tolist(),
        ],
    )


def test_histograms_by_reader_pdg(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, 211),
            (0.7, 0.3, 0.3, -211),
            (0.9, 0.9, 0.4, -211),
            (0.4, 0.4, 0.5, 321),
            (0.8, 0.2, 0.5, 321),
            (0.2, 0.2, 0.5, 999),
        ],
    )

    request = histogram_request(
        ["px", "py"],
        [
            brass.RegularAxis(2, 0.0, 1.0),
            brass.RegularAxis(2, 0.0, 1.0),
        ],
        histogram_group_by("pdg", [211, -211, 321]),
    )

    hists = brass.histogram(make_reader(path), request)

    assert set(hists.keys()) == {211, -211, 321}

    assert np.array_equal(
        hists[211].values,
        np.array(
            [
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        ),
    )

    assert np.array_equal(
        hists[-211].values,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )

    assert np.array_equal(
        hists[321].values,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    )

    assert hists[211].values.sum() == 2.0
    assert hists[-211].values.sum() == 2.0
    assert hists[321].values.sum() == 2.0


def test_histograms_by_particles_pdg(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, 211),
            (0.7, 0.3, 0.3, -211),
            (0.9, 0.9, 0.4, -211),
            (0.4, 0.4, 0.5, 321),
            (0.8, 0.2, 0.5, 321),
            (0.2, 0.2, 0.5, 999),
        ],
    )

    block = first_particle_block(path)

    request = histogram_request(
        ["px", "py"],
        [
            brass.RegularAxis(2, 0.0, 1.0),
            brass.RegularAxis(2, 0.0, 1.0),
        ],
        histogram_group_by("pdg", [211, -211, 321]),
    )

    hists = brass.histogram(block.particles, request)

    assert set(hists.keys()) == {211, -211, 321}

    assert np.array_equal(
        hists[211].values,
        np.array(
            [
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        ),
    )

    assert np.array_equal(
        hists[-211].values,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )

    assert np.array_equal(
        hists[321].values,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    )


def test_histograms_by_rejects_non_integer_group_column(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, -211),
        ],
    )

    request = histogram_request(
        ["px"],
        [brass.RegularAxis(2, 0.0, 1.0)],
        histogram_group_by("px", [0]),
    )

    try:
        brass.histogram(make_reader(path), request)
    except RuntimeError as error:
        assert "quantity is not int32: px" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_histogram_rejects_quantity_axis_mismatch(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.2, 0.3, 211),
        ],
    )

    request = histogram_request(
        ["px", "py"],
        [brass.RegularAxis(2, 0.0, 1.0)],
    )

    try:
        brass.histogram(make_reader(path), request)
    except RuntimeError as error:
        assert "number of histogram quantities must match number of axes" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_multiple_histograms_from_reader_one_pass(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, 211),
            (0.7, 0.3, 0.3, -211),
            (0.9, 0.9, 0.4, -211),
            (1.2, 0.5, 0.5, 321),
        ],
    )

    px_request = histogram_request(
        ["px"],
        [brass.RegularAxis(2, 0.0, 1.0)],
    )

    py_request = histogram_request(
        ["py"],
        [brass.RegularAxis(2, 0.0, 1.0)],
    )

    h_px, h_py = brass.histograms(
        make_reader(path),
        [px_request, py_request],
    )

    assert h_px.shape == [2]
    assert h_px.values.shape == (2,)
    assert np.array_equal(h_px.values, np.array([2.0, 2.0]))
    assert_edges_equal(h_px, [[0.0, 0.5, 1.0]])

    assert h_py.shape == [2]
    assert h_py.values.shape == (2,)
    assert np.array_equal(h_py.values, np.array([2.0, 3.0]))
    assert_edges_equal(h_py, [[0.0, 0.5, 1.0]])


def test_multiple_2d_histograms_from_reader_one_pass(tmp_path):
    path = tmp_path / "particles.bin"

    make_file(
        path,
        [
            (0.1, 0.1, 0.1, 211),
            (0.2, 0.8, 0.2, 211),
            (0.7, 0.3, 0.8, -211),
            (0.9, 0.9, 0.9, -211),
            (1.2, 0.5, 0.5, 321),
        ],
    )

    px_py_request = histogram_request(
        ["px", "py"],
        [
            brass.RegularAxis(2, 0.0, 1.0),
            brass.RegularAxis(2, 0.0, 1.0),
        ],
    )

    py_pz_request = histogram_request(
        ["py", "pz"],
        [
            brass.RegularAxis(2, 0.0, 1.0),
            brass.RegularAxis(2, 0.0, 1.0),
        ],
    )

    h_px_py, h_py_pz = brass.histograms(
        make_reader(path),
        [px_py_request, py_pz_request],
    )

    expected_px_py = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )

    expected_py_pz = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )

    assert h_px_py.shape == [2, 2]
    assert h_px_py.values.shape == (2, 2)
    assert np.array_equal(h_px_py.values, expected_px_py)

    assert h_py_pz.shape == [2, 2]
    assert h_py_pz.values.shape == (2, 2)
    assert np.array_equal(h_py_pz.values, expected_py_pz)
