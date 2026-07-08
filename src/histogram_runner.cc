#include "histogram_runner.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace brass {

void fill_histogram(
    const Particles& particles, const std::vector<std::string>& quantities,
    Histogram& hist,
    const std::optional<std::string>& particle_weight_quantity = std::nullopt,
    double block_weight = 1.0) {
    if (quantities.size() != hist.rank()) {
        throw std::runtime_error("histogram: quantity/axis dimension mismatch");
    }

    std::vector<double> values(quantities.size());

    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto row = particles.row(i);

        for (std::size_t d = 0; d < quantities.size(); ++d) {
            values[d] = particles.value(row, quantities[d]);
        }

        double weight = block_weight;

        if (particle_weight_quantity) {
            weight *= particles.value(row, *particle_weight_quantity);
        }

        hist.fill_values(values, weight);
    }
}

void fill_grouped_histogram(
    const Particles& particles, const std::vector<std::string>& quantities,
    const HistogramGroupBy& group_by,
    std::unordered_map<int32_t, Histogram>& hists,
    const std::optional<std::string>& particle_weight_quantity = std::nullopt,
    double block_weight = 1.0) {
    if (hists.empty()) {
        return;
    }

    const auto rank = hists.begin()->second.rank();

    if (quantities.size() != rank) {
        throw std::runtime_error("histogram: quantity/axis dimension mismatch");
    }

    const auto& group_quantity = particles.quantity(group_by.quantity);

    std::vector<double> values(quantities.size());

    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto row = particles.row(i);
        const auto group_value = row.get_int32(group_quantity);

        const auto it = hists.find(group_value);
        if (it == hists.end()) {
            continue;
        }

        for (std::size_t d = 0; d < quantities.size(); ++d) {
            values[d] = particles.value(row, quantities[d]);
        }

        double weight = block_weight;

        if (particle_weight_quantity) {
            weight *= particles.value(row, *particle_weight_quantity);
        }

        it->second.fill_values(values, weight);
    }
}

HistogramResult histogram_to_result(const Histogram& hist) {
    return HistogramResult{
        .values = hist.values(),
        .edges = hist.edges(),
        .shape = hist.shape(),
    };
}

GroupedHistogramResult grouped_histogram_to_result(
    const std::unordered_map<int32_t, Histogram>& hists) {
    GroupedHistogramResult results;

    for (const auto& [group, hist] : hists) {
        results.emplace(group, histogram_to_result(hist));
    }

    return results;
}

std::unordered_map<int32_t, Histogram> make_grouped_histograms(
    const std::vector<AxisConfig>& axes, const std::vector<int32_t>& groups) {
    std::unordered_map<int32_t, Histogram> hists;

    for (const auto group : groups) {
        hists.emplace(group, Histogram(axes));
    }

    return hists;
}

HistogramRun make_run(HistogramRequest request) {
    const auto axes = request.axes;

    if (request.group_by) {
        auto hists = make_grouped_histograms(axes, request.group_by->values);

        return HistogramRun{
            .request = std::move(request),
            .histograms = std::move(hists),
        };
    }

    return HistogramRun{
        .request = std::move(request),
        .histograms = Histogram(axes),
    };
}

HistogramRunner::HistogramRunner(BinaryReader& reader,
                                 std::vector<HistogramRequest> requests)
    : reader_(reader) {
    runs_.reserve(requests.size());

    for (auto& request : requests) {
        runs_.push_back(make_run(std::move(request)));
    }
}

void HistogramRunner::fill(const Particles& particles) {
    for (auto& run : runs_) {
        if (run.request.target != HistogramTarget::Particles) {
            continue;
        }

        if (run.request.group_by) {
            auto& hists = std::get<std::unordered_map<int32_t, Histogram>>(
                run.histograms);

            fill_grouped_histogram(particles, run.request.quantities,
                                   *run.request.group_by, hists,
                                   run.request.weight_quantity);
        } else {
            auto& hist = std::get<Histogram>(run.histograms);

            fill_histogram(particles, run.request.quantities, hist,
                           run.request.weight_quantity);
        }
    }
}
void HistogramRunner::fill(const Block& block) {
    std::visit(
        [this](const auto& b) {
            using T = std::decay_t<decltype(b)>;

            if constexpr (std::is_same_v<T, ParticleBlock>) {
                for (auto& run : runs_) {
                    if (run.request.target != HistogramTarget::Particles) {
                        continue;
                    }

                    if (run.request.group_by) {
                        auto& hists =
                            std::get<std::unordered_map<int32_t, Histogram>>(
                                run.histograms);

                        fill_grouped_histogram(b.particles,
                                               run.request.quantities,
                                               *run.request.group_by, hists,
                                               run.request.weight_quantity);
                    } else {
                        auto& hist = std::get<Histogram>(run.histograms);

                        fill_histogram(b.particles, run.request.quantities,
                                       hist, run.request.weight_quantity);
                    }
                }
            }

            if constexpr (std::is_same_v<T, EndBlock>) {
                fill_block_histograms(runs_, HistogramTarget::EndBlock, b,
                                      end_block_value);
            }

            if constexpr (std::is_same_v<T, InteractionBlock>) {
                fill_block_histograms(runs_, HistogramTarget::Interaction, b,
                                      interaction_value);

                for (auto& run : runs_) {
                    if (run.request.target != HistogramTarget::Incoming &&
                        run.request.target != HistogramTarget::Outgoing) {
                        continue;
                    }

                    const Particles& particles =
                        run.request.target == HistogramTarget::Incoming
                            ? b.incoming
                            : b.outgoing;

                    double block_weight = 1.0;
                    std::optional<std::string> particle_weight_quantity;

                    if (run.request.weight_quantity) {
                        const auto& weight_quantity =
                            *run.request.weight_quantity;

                        if (is_interaction_quantity(weight_quantity)) {
                            block_weight =
                                interaction_value(b, weight_quantity);
                        } else {
                            particle_weight_quantity = weight_quantity;
                        }
                    }

                    if (run.request.group_by) {
                        auto& hists =
                            std::get<std::unordered_map<int32_t, Histogram>>(
                                run.histograms);

                        fill_grouped_histogram(
                            particles, run.request.quantities,
                            *run.request.group_by, hists,
                            particle_weight_quantity, block_weight);
                    } else {
                        auto& hist = std::get<Histogram>(run.histograms);

                        fill_histogram(particles, run.request.quantities, hist,
                                       particle_weight_quantity, block_weight);
                    }
                }
            }
        },
        block);
}
void HistogramRunner::run() {
    while (true) {
        auto block = reader_.read();

        if (!block) {
            break;
        }

        fill(*block);
    }
}

HistogramBatchResult HistogramRunner::result() const {
    HistogramBatchResult results;
    results.reserve(runs_.size());

    for (const auto& run : runs_) {
        if (const auto* hist = std::get_if<Histogram>(&run.histograms)) {
            results.push_back(histogram_to_result(*hist));
        } else {
            const auto& hists =
                std::get<std::unordered_map<int32_t, Histogram>>(
                    run.histograms);

            results.push_back(grouped_histogram_to_result(hists));
        }
    }

    return results;
}

}  // namespace brass
