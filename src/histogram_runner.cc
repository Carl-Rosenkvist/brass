#include "histogram_runner.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

#include "binaryreader.h"
#include "histogram.h"
#include "quantity_expr.h"

namespace brass {

namespace {

std::vector<QuantityExpr> make_quantity_expressions(
    const std::vector<std::string>& histogram_quantities, std::size_t naxes,
    const std::string& caller) {
    if (histogram_quantities.empty()) {
        throw std::runtime_error(caller + ": no histogram quantities given");
    }

    if (histogram_quantities.size() != naxes) {
        throw std::runtime_error(
            caller +
            ": number of histogram quantities must match number of axes");
    }

    std::vector<QuantityExpr> expressions;
    expressions.reserve(histogram_quantities.size());

    for (const auto& name : histogram_quantities) {
        expressions.push_back(quantity_expr_from_name(name));
    }

    return expressions;
}

void fill_histogram_from_particles(
    Histogram& hist, const Particles& particles,
    const std::vector<QuantityExpr>& expressions) {
    std::vector<double> values(expressions.size());

    for (std::size_t i = 0; i < particles.size(); ++i) {
        bool valid = true;

        for (std::size_t d = 0; d < expressions.size(); ++d) {
            values[d] = evaluate_quantity(expressions[d], particles, i);

            if (!std::isfinite(values[d])) {
                valid = false;
                break;
            }
        }

        if (valid) {
            hist.fill_values(values);
        }
    }
}

void fill_histogram_from_reader(Histogram& hist, BinaryReader& reader,
                                const std::vector<QuantityExpr>& expressions) {
    while (true) {
        auto block = reader.read();

        if (!block.has_value()) {
            break;
        }

        if (!std::holds_alternative<ParticleBlock>(*block)) {
            continue;
        }

        fill_histogram_from_particles(
            hist, std::get<ParticleBlock>(*block).particles, expressions);
    }
}

std::unordered_map<int32_t, Histogram> make_grouped_histograms(
    const std::vector<AxisConfig>& axes,
    const std::vector<int32_t>& group_values) {
    std::unordered_map<int32_t, Histogram> histograms;
    histograms.reserve(group_values.size());

    for (const auto value : group_values) {
        histograms.emplace(value, Histogram(axes));
    }

    return histograms;
}

void fill_grouped_histograms_from_particles(
    std::unordered_map<int32_t, Histogram>& histograms,
    const Particles& particles, const std::vector<QuantityExpr>& expressions,
    const std::string& by) {
    const auto& group = particles.column<int32_t>(by);
    std::vector<double> values(expressions.size());

    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto it = histograms.find(group[i]);

        if (it == histograms.end()) {
            continue;
        }

        bool valid = true;

        for (std::size_t d = 0; d < expressions.size(); ++d) {
            values[d] = evaluate_quantity(expressions[d], particles, i);

            if (!std::isfinite(values[d])) {
                valid = false;
                break;
            }
        }

        if (valid) {
            it->second.fill_values(values);
        }
    }
}

void fill_grouped_histograms_from_reader(
    std::unordered_map<int32_t, Histogram>& histograms, BinaryReader& reader,
    const std::vector<QuantityExpr>& expressions, const std::string& by) {
    while (true) {
        auto block = reader.read();

        if (!block.has_value()) {
            break;
        }

        if (!std::holds_alternative<ParticleBlock>(*block)) {
            continue;
        }

        fill_grouped_histograms_from_particles(
            histograms, std::get<ParticleBlock>(*block).particles, expressions,
            by);
    }
}

HistogramResult make_result(const Histogram& hist) {
    return HistogramResult{
        .values = hist.values(),
        .edges = hist.edges(),
        .shape = hist.shape(),
    };
}

std::unordered_map<int32_t, HistogramResult> make_grouped_result(
    const std::unordered_map<int32_t, Histogram>& histograms) {
    std::unordered_map<int32_t, HistogramResult> out;
    out.reserve(histograms.size());

    for (const auto& [key, hist] : histograms) {
        out.emplace(key, make_result(hist));
    }

    return out;
}

}  // namespace

HistogramResult histogram_particles(
    const Particles& particles,
    const std::vector<std::string>& histogram_quantities,
    const std::vector<AxisConfig>& axes) {
    const auto expressions = make_quantity_expressions(
        histogram_quantities, axes.size(), "histogram_particles");

    Histogram hist(axes);
    fill_histogram_from_particles(hist, particles, expressions);

    return make_result(hist);
}

HistogramResult histogram_reader(
    BinaryReader& reader, const std::vector<std::string>& histogram_quantities,
    const std::vector<AxisConfig>& axes) {
    const auto expressions = make_quantity_expressions(
        histogram_quantities, axes.size(), "histogram_reader");

    Histogram hist(axes);
    fill_histogram_from_reader(hist, reader, expressions);

    return make_result(hist);
}

std::unordered_map<int32_t, HistogramResult> histograms_by_particles(
    const Particles& particles,
    const std::vector<std::string>& histogram_quantities,
    const std::vector<AxisConfig>& axes, const std::string& by,
    const std::vector<int32_t>& group_values) {
    const auto expressions = make_quantity_expressions(
        histogram_quantities, axes.size(), "histograms_by_particles");

    auto histograms = make_grouped_histograms(axes, group_values);

    fill_grouped_histograms_from_particles(histograms, particles, expressions,
                                           by);

    return make_grouped_result(histograms);
}

std::unordered_map<int32_t, HistogramResult> histograms_by_reader(
    BinaryReader& reader, const std::vector<std::string>& histogram_quantities,
    const std::vector<AxisConfig>& axes, const std::string& by,
    const std::vector<int32_t>& group_values) {
    const auto expressions = make_quantity_expressions(
        histogram_quantities, axes.size(), "histograms_by_reader");

    auto histograms = make_grouped_histograms(axes, group_values);

    fill_grouped_histograms_from_reader(histograms, reader, expressions, by);

    return make_grouped_result(histograms);
}

}  // namespace brass
