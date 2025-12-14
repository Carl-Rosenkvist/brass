#include "histogram.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace brass {

Histogram::Histogram(std::initializer_list<std::string> axis_names,
                     std::initializer_list<double> bin_widths,
                     std::size_t estimated_max_bin_size_1D)
    : dim(axis_names.size()),
      axis_names(axis_names),
      bin_widths(bin_widths),
      estimated_max_bin_size_1D(estimated_max_bin_size_1D) {
    if (bin_widths.size() != dim) {
        throw std::runtime_error("bin_widths size mismatch");
    }

    for (double w : bin_widths) {
        if (w <= 0.0) {
            throw std::runtime_error("bin width must be positive");
        }
    }

    std::size_t estimated_size = 1;
    for (std::size_t d = 0; d < dim; ++d) {
        estimated_size *= estimated_max_bin_size_1D;
    }
    bins.reserve(estimated_size);
}

std::ptrdiff_t Histogram::logical_index(double value, std::size_t d) const {
    return static_cast<std::ptrdiff_t>(std::floor(value / bin_widths[d]));
}

}  // namespace brass
