#include "histogram.h"

#include <boost/histogram.hpp>
#include <stdexcept>
#include <utility>

namespace brass {

namespace bh = boost::histogram;

Histogram::Histogram(std::vector<AxisConfig> axes) : hist_(make_axes(axes)) {}

histogram_detail::BoostAxis Histogram::make_axis(const AxisConfig& axis) {
    return std::visit(
        [](const auto& cfg) -> histogram_detail::BoostAxis {
            using T = std::decay_t<decltype(cfg)>;

            if constexpr (std::is_same_v<T, RegularAxis>) {
                if (cfg.bins <= 0) {
                    throw std::runtime_error(
                        "Histogram: regular axis bins must be positive");
                }

                if (!(cfg.lower < cfg.upper)) {
                    throw std::runtime_error(
                        "Histogram: regular axis lower must be smaller than "
                        "upper");
                }

                return bh::axis::regular<>(cfg.bins, cfg.lower, cfg.upper);
            }

            else if constexpr (std::is_same_v<T, VariableAxis>) {
                if (cfg.edges.size() < 2) {
                    throw std::runtime_error(
                        "Histogram: variable axis needs at least two edges");
                }

                return bh::axis::variable<>(cfg.edges);
            }

            else if constexpr (std::is_same_v<T, IntegerAxis>) {
                if (!(cfg.lower < cfg.upper)) {
                    throw std::runtime_error(
                        "Histogram: integer axis lower must be smaller than "
                        "upper");
                }

                return bh::axis::integer<>(cfg.lower, cfg.upper);
            }
        },
        axis);
}

histogram_detail::BoostAxes Histogram::make_axes(
    const std::vector<AxisConfig>& axes) {
    if (axes.empty()) {
        throw std::runtime_error("Histogram: at least one axis required");
    }

    histogram_detail::BoostAxes out;
    out.reserve(axes.size());

    for (const auto& axis : axes) {
        out.push_back(make_axis(axis));
    }

    return out;
}

std::size_t Histogram::rank() const { return hist_.rank(); }

std::size_t Histogram::size() const { return hist_.size(); }

std::vector<double> Histogram::values() const {
    std::vector<double> out;
    out.reserve(hist_.size());

    for (auto&& cell : bh::indexed(hist_)) {
        out.push_back(static_cast<double>(*cell));
    }

    return out;
}

std::vector<std::size_t> Histogram::shape() const {
    std::vector<std::size_t> out;
    out.reserve(rank());

    for (std::size_t i = 0; i < rank(); ++i) {
        out.push_back(hist_.axis(i).size());
    }

    return out;
}

std::vector<std::vector<double>> Histogram::edges() const {
    std::vector<std::vector<double>> out;
    out.reserve(rank());

    for (std::size_t iaxis = 0; iaxis < rank(); ++iaxis) {
        const auto& axis = hist_.axis(iaxis);

        std::vector<double> axis_edges;
        axis_edges.reserve(axis.size() + 1);

        for (auto&& bin : axis) {
            axis_edges.push_back(bin.lower());
        }

        if (axis.size() > 0) {
            axis_edges.push_back(axis.value(axis.size()));
        }

        out.push_back(std::move(axis_edges));
    }

    return out;
}
void Histogram::fill_values(const std::vector<double>& values) {
    if (values.size() != rank()) {
        throw std::runtime_error("Histogram: fill dimension mismatch");
    }

    switch (values.size()) {
        case 1:
            hist_(values[0]);
            return;

        case 2:
            hist_(values[0], values[1]);
            return;

        case 3:
            hist_(values[0], values[1], values[2]);
            return;

        case 4:
            hist_(values[0], values[1], values[2], values[3]);
            return;

        default:
            throw std::runtime_error(
                "Histogram: runtime fill supports only 1D to 4D for now");
    }
}

}  // namespace brass
