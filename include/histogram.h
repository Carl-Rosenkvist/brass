#pragma once

#include <cstddef>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

namespace brass {

class Histogram {
   public:
    Histogram() = delete;

    explicit Histogram(std::initializer_list<std::string> axis_names,
                       std::initializer_list<double> bin_widths,
                       std::size_t estimated_max_bin_size_1D = 1);

    void fill(std::span<const double> values, double weight = 1.0);

   private:
    std::size_t dim;
    std::size_t estimated_max_bin_size_1D;
    std::vector<std::string> axis_names;
    std::vector<double> bin_widths;

    std::vector<std::size_t> nbins;
    std::vector<std::size_t> strides;
    std::vector<double> bins;

    std::vector<std::ptrdiff_t> min_index;
    std::vector<std::ptrdiff_t> max_index;

    std::ptrdiff_t logical_index(double value, std::size_t d) const;

    void unflatten_index(std::size_t flat,
                         std::span<std::size_t> indices) const;
    void ensure_contains(std::span<const std::ptrdiff_t> logical_indices);
    std::size_t flatten_index(std::span<const std::size_t> indices) const;
};
}  // namespace brass
