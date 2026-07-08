import struct

import numpy as np
import pytest

import brass


Q = ["px", "py", "pz", "pdg"]
P4 = ["p0", "px", "py", "pz"]

TYPE = {
    "p0": "<d",
    "px": "<d",
    "py": "<d",
    "pz": "<d",
    "pdg": "<i",
}

DEFAULT_EVENT_NUMBER = 1
DEFAULT_ENSEMBLE_NUMBER = 0
DEFAULT_END_IMPACT_PARAMETER = 0.0
DEFAULT_END_EMPTY = False

INTERACTION_QUANTITIES = ["n_in", "n_out", "rho", "sigma", "sigma_p", "process"]


def request(names, axes, *, target=None, group_by=None, weight_quantity=None):
    r = brass.HistogramRequest()
    r.quantities = names
    r.axes = axes
    r.group_by = group_by
    r.weight_quantity = weight_quantity

    if target is not None:
        r.target = target

    return r


def group_by(name, values):
    return brass.HistogramGroupBy(name, values)


def axis(n=2, low=0.0, high=1.0):
    return brass.RegularAxis(n, low, high)


def bins(n=2, low=0.0, high=1.0):
    return np.linspace(low, high, n + 1)


def rows(data):
    return np.array(data, dtype=object)


def write_header(f, version=b"SMASH-3.1", format_version=9, format_variant=1):
    f.write(b"SMSH")
    f.write(struct.pack("<HHI", format_version, format_variant, len(version)))
    f.write(version)


def write_end(
    f,
    *,
    event_number=DEFAULT_EVENT_NUMBER,
    ensemble_number=DEFAULT_ENSEMBLE_NUMBER,
    impact_parameter=DEFAULT_END_IMPACT_PARAMETER,
    empty=DEFAULT_END_EMPTY,
):
    f.write(b"f")
    f.write(
        struct.pack("<Iid?", event_number, ensemble_number, impact_parameter, empty)
    )


def write_particle_rows(f, rows_, quantities):
    for row in rows_:
        for value, quantity in zip(row, quantities):
            f.write(struct.pack(TYPE[quantity], value))


def write_particles(
    f,
    rows_,
    quantities=Q,
    *,
    event_number=DEFAULT_EVENT_NUMBER,
    ensemble_number=DEFAULT_ENSEMBLE_NUMBER,
):
    f.write(b"p")
    f.write(struct.pack("<iiI", event_number, ensemble_number, len(rows_)))
    write_particle_rows(f, rows_, quantities)


def write_particle_file(path, rows_, quantities=Q):
    with open(path, "wb") as f:
        write_header(f)
        write_particles(f, rows_, quantities)
        write_end(f)


def interaction(incoming, outgoing, *, rho, sigma, sigma_p=0.0, process=0):
    return {
        "incoming": rows(incoming),
        "outgoing": rows(outgoing),
        "rho": rho,
        "sigma": sigma,
        "sigma_p": sigma_p,
        "process": process,
    }


def interaction_value(interaction_, name):
    if name == "n_in":
        return len(interaction_["incoming"])

    if name == "n_out":
        return len(interaction_["outgoing"])

    return interaction_[name]


def interaction_table(interactions, names):
    return np.array(
        [
            [interaction_value(interaction_, name) for name in names]
            for interaction_ in interactions
        ],
        dtype=float,
    )


def particle_table_from_interactions(interactions, which):
    return np.vstack([interaction_[which] for interaction_ in interactions])


def repeated_interaction_weights(interactions, which, weight_quantity):
    return np.concatenate(
        [
            np.full(
                len(interaction_[which]),
                interaction_value(interaction_, weight_quantity),
            )
            for interaction_ in interactions
        ]
    )


def write_interaction(f, interaction_, quantities=Q):
    incoming = interaction_["incoming"]
    outgoing = interaction_["outgoing"]

    f.write(b"i")
    f.write(struct.pack("<ii", len(incoming), len(outgoing)))
    f.write(
        struct.pack(
            "<dddi",
            interaction_["rho"],
            interaction_["sigma"],
            interaction_["sigma_p"],
            interaction_["process"],
        )
    )

    write_particle_rows(f, incoming, quantities)
    write_particle_rows(f, outgoing, quantities)


def write_interaction_file(path, interactions, quantities=Q):
    with open(path, "wb") as f:
        write_header(f)

        for interaction_ in interactions:
            write_interaction(f, interaction_, quantities)

        write_end(f)


def run(path, requests, read=Q):
    reader = brass.BinaryReader(str(path), read)
    runner = brass.HistogramRunner(reader, requests)
    runner.run()
    return runner.result()


def hist(
    path,
    names,
    axes,
    *,
    read=Q,
    target=None,
    group_by=None,
    weight_quantity=None,
):
    return run(
        path,
        [
            request(
                names,
                axes,
                target=target,
                group_by=group_by,
                weight_quantity=weight_quantity,
            )
        ],
        read,
    )[0]


def values(rows_, read, names):
    cols = {name: rows_[:, i].astype(float) for i, name in enumerate(read)}

    def value(name):
        if name == "pt":
            return np.hypot(value("px"), value("py"))

        if name == "p":
            return np.sqrt(value("px") ** 2 + value("py") ** 2 + value("pz") ** 2)

        if name == "mass":
            return np.sqrt(np.maximum(value("p0") ** 2 - value("p") ** 2, 0.0))

        if name == "mt":
            return np.hypot(value("mass"), value("pt"))

        if name == "y_rap":
            return 0.5 * np.log(
                (value("p0") + value("pz")) / (value("p0") - value("pz"))
            )

        return cols[name]

    return np.column_stack([value(name) for name in names])


def expected(rows_, read, names, edges, weight_quantity=None, weights=None):
    if weights is None and weight_quantity is not None:
        weights = values(rows_, read, [weight_quantity])[:, 0]

    return np.histogramdd(
        values(rows_, read, names),
        bins=edges,
        weights=weights,
    )[0]


def expected_interaction(interactions, names, edges, weight_quantity=None):
    weights = None

    if weight_quantity is not None:
        weights = interaction_table(interactions, [weight_quantity])[:, 0]

    return np.histogramdd(
        interaction_table(interactions, names),
        bins=edges,
        weights=weights,
    )[0]


def check(result, expected_values):
    assert result.shape == list(expected_values.shape)
    assert np.array_equal(result.values, expected_values)


def particle_rows():
    return rows(
        [
            [0.1, 0.1, 0.1, 211],
            [0.2, 0.8, 0.2, 211],
            [0.7, 0.3, 0.8, -211],
            [0.9, 0.9, 0.9, -211],
            [1.2, 0.5, 0.5, 321],
        ]
    )


def grouped_particle_rows():
    return rows(
        [
            [0.1, 0.1, 0.1, 211],
            [0.2, 0.8, 0.2, 211],
            [0.7, 0.3, 0.3, -211],
            [0.9, 0.9, 0.4, -211],
            [0.4, 0.4, 0.5, 321],
            [0.8, 0.2, 0.5, 321],
            [0.2, 0.2, 0.5, 999],
        ]
    )


def interaction_rows():
    return [
        interaction(
            incoming=[
                [0.1, 0.1, 0.1, 211],
            ],
            outgoing=[
                [0.1, 0.1, 0.1, 211],
                [0.6, 0.1, 0.1, -211],
            ],
            rho=0.25,
            sigma=2.0,
            sigma_p=0.1,
            process=1,
        ),
        interaction(
            incoming=[
                [0.3, 0.3, 0.3, 211],
            ],
            outgoing=[
                [0.2, 0.1, 0.1, 211],
                [0.8, 0.1, 0.1, -211],
            ],
            rho=0.75,
            sigma=3.0,
            sigma_p=0.2,
            process=2,
        ),
    ]


def test_particle_histograms(tmp_path):
    data = particle_rows()
    path = tmp_path / "particles.bin"
    write_particle_file(path, data)

    cases = [
        (["px"], [axis()], [bins()]),
        (["px", "py"], [axis(), axis()], [bins(), bins()]),
        (["px", "py", "pz"], [axis(), axis(), axis()], [bins(), bins(), bins()]),
        (["pt"], [axis()], [bins()]),
    ]

    for names, axes, edges in cases:
        check(
            hist(path, names, axes),
            expected(data, Q, names, edges),
        )


def test_weighted_particle_histogram(tmp_path):
    data = particle_rows()
    path = tmp_path / "particles.bin"
    write_particle_file(path, data)

    check(
        hist(path, ["px"], [axis()], weight_quantity="pz"),
        expected(data, Q, ["px"], [bins()], weight_quantity="pz"),
    )


def test_p4_histograms(tmp_path):
    data = np.array(
        [
            [1.9, 0.3, 0.4, 0.0],
            [3.0, 0.6, 0.8, 1.0],
            [3.0, 0.1, 0.1, -1.0],
            [1.0, 0.1, 0.1, 1.0],
        ],
        dtype=float,
    )

    path = tmp_path / "p4.bin"
    write_particle_file(path, data, P4)

    names = ["mt", "y_rap"]
    axes = [brass.RegularAxis(3, 0.0, 3.0), brass.RegularAxis(2, -1.0, 1.0)]
    edges = [bins(3, 0.0, 3.0), bins(2, -1.0, 1.0)]

    check(
        hist(path, names, axes, read=P4),
        expected(data, P4, names, edges),
    )


def test_group_by(tmp_path):
    data = grouped_particle_rows()
    path = tmp_path / "particles.bin"
    write_particle_file(path, data)

    groups = [211, -211, 321]

    result = hist(
        path,
        ["px", "py"],
        [axis(), axis()],
        group_by=group_by("pdg", groups),
    )

    for pdg in groups:
        mask = data[:, Q.index("pdg")] == pdg
        check(
            result[pdg],
            expected(data[mask], Q, ["px", "py"], [bins(), bins()]),
        )


def test_weighted_group_by(tmp_path):
    data = grouped_particle_rows()
    path = tmp_path / "particles.bin"
    write_particle_file(path, data)

    groups = [211, -211, 321]

    result = hist(
        path,
        ["px", "py"],
        [axis(), axis()],
        group_by=group_by("pdg", groups),
        weight_quantity="pz",
    )

    for pdg in groups:
        mask = data[:, Q.index("pdg")] == pdg

        check(
            result[pdg],
            expected(
                data[mask],
                Q,
                ["px", "py"],
                [bins(), bins()],
                weight_quantity="pz",
            ),
        )


def test_interaction_histogram_with_interaction_weight(tmp_path):
    interactions = interaction_rows()
    path = tmp_path / "interactions.bin"
    write_interaction_file(path, interactions)

    check(
        hist(
            path,
            ["rho"],
            [axis()],
            target=brass.HistogramTarget.Interaction,
            weight_quantity="sigma",
        ),
        expected_interaction(
            interactions,
            ["rho"],
            [bins()],
            weight_quantity="sigma",
        ),
    )


def test_outgoing_particle_histogram_with_interaction_weight(tmp_path):
    interactions = interaction_rows()
    path = tmp_path / "interactions.bin"
    write_interaction_file(path, interactions)

    particle_data = particle_table_from_interactions(interactions, "outgoing")
    weights = repeated_interaction_weights(interactions, "outgoing", "sigma")

    check(
        hist(
            path,
            ["px"],
            [axis()],
            target=brass.HistogramTarget.Outgoing,
            weight_quantity="sigma",
        ),
        expected(
            particle_data,
            Q,
            ["px"],
            [bins()],
            weights=weights,
        ),
    )


def test_grouped_outgoing_particle_histogram_with_interaction_weight(tmp_path):
    interactions = interaction_rows()
    path = tmp_path / "interactions.bin"
    write_interaction_file(path, interactions)

    particle_data = particle_table_from_interactions(interactions, "outgoing")
    weights = repeated_interaction_weights(interactions, "outgoing", "sigma")
    groups = [211, -211]

    result = hist(
        path,
        ["px"],
        [axis()],
        target=brass.HistogramTarget.Outgoing,
        group_by=group_by("pdg", groups),
        weight_quantity="sigma",
    )

    for pdg in groups:
        mask = particle_data[:, Q.index("pdg")] == pdg

        check(
            result[pdg],
            expected(
                particle_data[mask],
                Q,
                ["px"],
                [bins()],
                weights=weights[mask],
            ),
        )


def test_multiple_histograms_one_pass(tmp_path):
    data = particle_rows()
    path = tmp_path / "particles.bin"
    write_particle_file(path, data)

    cases = [
        (["px"], [axis()], [bins()]),
        (["py"], [axis()], [bins()]),
        (["px", "py"], [axis(), axis()], [bins(), bins()]),
    ]

    results = run(path, [request(names, axes) for names, axes, _ in cases])

    for result, (names, _, edges) in zip(results, cases):
        check(result, expected(data, Q, names, edges))


def test_rejections(tmp_path):
    data = rows([[0.1, 0.2, 0.3, 211]])
    path = tmp_path / "particles.bin"
    write_particle_file(path, data)

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        hist(path, ["px", "py"], [axis()])

    with pytest.raises(RuntimeError, match="quantity is not int32: px"):
        hist(
            path,
            ["px"],
            [axis()],
            group_by=group_by("px", [0]),
        )

    with pytest.raises(RuntimeError):
        hist(
            path,
            ["px"],
            [axis()],
            weight_quantity="does_not_exist",
        )
