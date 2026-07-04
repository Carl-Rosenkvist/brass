#include "histogram.h"

#include <stdexcept>
#include <vector>

#include "doctest.h"

TEST_CASE("Histogram fills one-dimensional regular axis") {
    brass::Histogram h(
        brass::RegularAxis{.bins = 5, .lower = 0.0, .upper = 1.0});

    h.fill(0.1);
    h.fill(0.2);
    h.fill(0.9);

    CHECK(h.rank() == 1);
    CHECK(h.shape() == std::vector<std::size_t>{5});

    const auto values = h.values();

    REQUIRE(values.size() == 5);
    CHECK(values[0] == 1.0);
    CHECK(values[1] == 1.0);
    CHECK(values[2] == 0.0);
    CHECK(values[3] == 0.0);
    CHECK(values[4] == 1.0);

    const auto edges = h.edges();

    REQUIRE(edges.size() == 1);
    CHECK(edges[0] == std::vector<double>{0.0, 0.2, 0.4, 0.6, 0.8, 1.0});
}

TEST_CASE("Histogram fills two-dimensional regular axes") {
    brass::Histogram h(
        brass::RegularAxis{.bins = 2, .lower = 0.0, .upper = 1.0},
        brass::RegularAxis{.bins = 2, .lower = -1.0, .upper = 1.0});

    h.fill(0.25, -0.5);
    h.fill(0.75, 0.5);

    CHECK(h.rank() == 2);
    CHECK(h.shape() == std::vector<std::size_t>{2, 2});

    const auto values = h.values();

    REQUIRE(values.size() == 4);

    double total = 0.0;
    for (const double value : values) {
        total += value;
    }

    CHECK(total == 2.0);
}

TEST_CASE("Histogram rejects fill dimension mismatch") {
    brass::Histogram h(
        brass::RegularAxis{.bins = 5, .lower = 0.0, .upper = 1.0});

    CHECK_THROWS_AS(h.fill(0.1, 0.2), std::runtime_error);
}
