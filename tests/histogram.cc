#include "histogram.h"

#include "doctest.h"

using brass::Histogram;

TEST_CASE("Histogram constructor: valid input") {
    REQUIRE_NOTHROW(Histogram({"pt", "y"}, {0.1, 0.2}));
}

TEST_CASE("Histogram constructor: valid input with optional arguments") {
    REQUIRE_NOTHROW(Histogram({"pt", "y"}, {0.1, 0.2}, 100));
}
