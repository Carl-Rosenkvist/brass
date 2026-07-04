#ifndef BRASS_HISTOGRAM_H
#define BRASS_HISTOGRAM_H

#include <boost/histogram.hpp>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace brass {

namespace histogram_detail {
namespace bh = boost::histogram;

using BoostAxis = bh::axis::variant<bh::axis::regular<>, bh::axis::variable<>,
                                    bh::axis::integer<>>;

using BoostAxes = std::vector<BoostAxis>;
using BoostHistogram = bh::histogram<BoostAxes>;
}  // namespace histogram_detail

struct RegularAxis {
    int bins = 0;
    double lower = 0.0;
    double upper = 0.0;
};

struct VariableAxis {
    std::vector<double> edges;
};

struct IntegerAxis {
    int lower = 0;
    int upper = 0;
};

using AxisConfig = std::variant<RegularAxis, VariableAxis, IntegerAxis>;

class Histogram {
   public:
    Histogram() = default;

    explicit Histogram(std::vector<AxisConfig> axes);

    template <class... AxisConfigs>
    explicit Histogram(AxisConfigs&&... axes)
        : Histogram(std::vector<AxisConfig>{
              AxisConfig{std::forward<AxisConfigs>(axes)}...}) {}

    template <class... Args>
    void fill(Args&&... args) {
        if (sizeof...(Args) != rank()) {
            throw std::runtime_error("Histogram: fill dimension mismatch");
        }

        hist_(std::forward<Args>(args)...);
    }
    void fill_values(const std::vector<double>& values);
    std::size_t rank() const;
    std::size_t size() const;

    std::vector<double> values() const;
    std::vector<std::vector<double>> edges() const;
    std::vector<std::size_t> shape() const;

   private:
    static histogram_detail::BoostAxis make_axis(const AxisConfig& axis);
    static histogram_detail::BoostAxes make_axes(
        const std::vector<AxisConfig>& axes);

    histogram_detail::BoostHistogram hist_;
};

}  // namespace brass

#endif  // BRASS_HISTOGRAM_H
