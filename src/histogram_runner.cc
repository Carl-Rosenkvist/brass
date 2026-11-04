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

namespace {

template <class Q, class Parse>
std::vector<Q> parse_all(const std::vector<std::string>& names, Parse parse) {
    std::vector<Q> out;
    out.reserve(names.size());

    for (const auto& name : names) {
        out.push_back(parse(name));
    }

    return out;
}

void fill_from_buffer(Histogram& hist, const FillBuffer& buffer,
                      bool weighted) {
    if (weighted) {
        hist.fill_columns(buffer.columns, buffer.weights);
    } else {
        hist.fill_columns(buffer.columns);
    }
}

// Compute the axis values (and weights) of every particle in the block.
// Returns whether the entries are weighted.
bool compute_block(const Particles& particles,
                   const std::vector<Quantity>& quantities, FillBuffer& buffer,
                   const std::optional<Quantity>& particle_weight,
                   double block_weight) {
    buffer.columns.resize(quantities.size());

    for (std::size_t d = 0; d < quantities.size(); ++d) {
        particles.compute(quantities[d], buffer.columns[d]);
    }

    if (particle_weight) {
        particles.compute(*particle_weight, buffer.weights);

        if (block_weight != 1.0) {
            for (auto& weight : buffer.weights) {
                weight *= block_weight;
            }
        }

        return true;
    }

    if (block_weight != 1.0) {
        buffer.weights.assign(particles.size(), block_weight);
        return true;
    }

    buffer.weights.clear();
    return false;
}

void fill_histogram(const Particles& particles,
                    const std::vector<Quantity>& quantities, Histogram& hist,
                    FillBuffer& buffer,
                    const std::optional<Quantity>& particle_weight,
                    double block_weight) {
    if (quantities.size() != hist.rank()) {
        throw std::runtime_error("histogram: quantity/axis dimension mismatch");
    }

    const bool weighted = compute_block(particles, quantities, buffer,
                                        particle_weight, block_weight);

    fill_from_buffer(hist, buffer, weighted);
}

void fill_grouped_histogram(
    const Particles& particles, const std::vector<Quantity>& quantities,
    Quantity group_by, std::unordered_map<int32_t, Histogram>& hists,
    FillBuffer& block, std::unordered_map<int32_t, FillBuffer>& buffers,
    const std::optional<Quantity>& particle_weight, double block_weight) {
    if (hists.empty()) {
        return;
    }

    const auto rank = hists.begin()->second.rank();

    if (quantities.size() != rank) {
        throw std::runtime_error("histogram: quantity/axis dimension mismatch");
    }

    const auto& group_quantity = particles.quantity(group_by);

    const bool weighted = compute_block(particles, quantities, block,
                                        particle_weight, block_weight);

    for (auto& [group, buffer] : buffers) {
        buffer.reset(rank);
    }

    // Distribute the block's entries to their groups.
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto group_value = particles.row(i).get_int32(group_quantity);

        const auto it = buffers.find(group_value);
        if (it == buffers.end()) {
            continue;
        }

        auto& buffer = it->second;

        for (std::size_t d = 0; d < rank; ++d) {
            buffer.columns[d].push_back(block.columns[d][i]);
        }

        if (weighted) {
            buffer.weights.push_back(block.weights[i]);
        }
    }

    for (auto& [group, buffer] : buffers) {
        if (buffer.size() > 0) {
            fill_from_buffer(hists.at(group), buffer, weighted);
        }
    }
}

template <class BlockType, class Q, class Value>
void fill_block_histogram(HistogramRun& run, const BlockType& block,
                          const std::vector<Q>& quantities,
                          const std::optional<Q>& weight_quantity,
                          Value value) {
    auto& hist = std::get<Histogram>(run.histograms);

    if (quantities.size() != hist.rank()) {
        throw std::runtime_error("histogram: quantity/axis dimension mismatch");
    }

    std::vector<double> values(quantities.size());

    for (std::size_t d = 0; d < quantities.size(); ++d) {
        values[d] = value(block, quantities[d]);
    }

    const double weight =
        weight_quantity ? value(block, *weight_quantity) : 1.0;

    hist.fill_values(values, weight);
}

}  // namespace

ResolvedRequest resolve_request(const HistogramRequest& request) {
    ResolvedRequest out;
    out.target = request.target;

    const auto& names = request.quantities;
    const auto& weight = request.weight_quantity;

    switch (request.target) {
        case HistogramTarget::Particles:
            out.particle_quantities = parse_all<Quantity>(names, parse_quantity);
            if (weight) {
                out.particle_weight = parse_quantity(*weight);
            }
            break;

        case HistogramTarget::Incoming:
        case HistogramTarget::Outgoing:
            out.particle_quantities = parse_all<Quantity>(names, parse_quantity);
            if (weight) {
                // A weight can come from the interaction (e.g. sigma) or
                // from each particle.
                out.interaction_weight = try_parse_interaction_quantity(*weight);
                if (!out.interaction_weight) {
                    out.particle_weight = parse_quantity(*weight);
                }
            }
            break;

        case HistogramTarget::Interaction:
            out.interaction_quantities = parse_all<InteractionQuantity>(
                names, parse_interaction_quantity);
            if (weight) {
                out.interaction_weight = parse_interaction_quantity(*weight);
            }
            break;

        case HistogramTarget::EndBlock:
            out.end_block_quantities = parse_all<EndBlockQuantity>(
                names, parse_end_block_quantity);
            if (weight) {
                out.end_block_weight = parse_end_block_quantity(*weight);
            }
            break;

        case HistogramTarget::Header:
            out.header_quantities =
                parse_all<HeaderQuantity>(names, parse_header_quantity);
            if (weight) {
                out.header_weight = parse_header_quantity(*weight);
            }
            break;
    }

    if (request.group_by) {
        const bool particle_target =
            request.target == HistogramTarget::Particles ||
            request.target == HistogramTarget::Incoming ||
            request.target == HistogramTarget::Outgoing;

        if (!particle_target) {
            throw std::runtime_error(
                "group_by is not supported for block histograms");
        }

        const auto q = parse_quantity(request.group_by->quantity);

        if (!is_stored_quantity(q) || quantity_type(q) != QuantityType::Int32) {
            throw std::runtime_error("group_by quantity is not int32: " +
                                     request.group_by->quantity);
        }

        out.group_by = q;
    }

    return out;
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

HistogramRun make_run(const HistogramRequest& request) {
    auto resolved = resolve_request(request);

    if (request.group_by) {
        HistogramRun run{
            .request = std::move(resolved),
            .histograms =
                make_grouped_histograms(request.axes, request.group_by->values),
        };

        for (const auto group : request.group_by->values) {
            run.group_buffers.try_emplace(group);
        }

        return run;
    }

    return HistogramRun{
        .request = std::move(resolved),
        .histograms = Histogram(request.axes),
    };
}

HistogramRunner::HistogramRunner(BinaryReader& reader,
                                 std::vector<HistogramRequest> requests)
    : reader_(reader) {
    runs_.reserve(requests.size());

    for (const auto& request : requests) {
        runs_.push_back(make_run(request));
    }
}

void HistogramRunner::fill_particles(HistogramRun& run,
                                     const Particles& particles,
                                     double block_weight) {
    const auto& request = run.request;

    if (request.group_by) {
        auto& hists =
            std::get<std::unordered_map<int32_t, Histogram>>(run.histograms);

        fill_grouped_histogram(particles, request.particle_quantities,
                               *request.group_by, hists, run.buffer,
                               run.group_buffers,
                               request.particle_weight, block_weight);
    } else {
        fill_histogram(particles, request.particle_quantities,
                       std::get<Histogram>(run.histograms), run.buffer,
                       request.particle_weight, block_weight);
    }
}

void HistogramRunner::fill(const Particles& particles) {
    for (auto& run : runs_) {
        if (run.request.target == HistogramTarget::Particles) {
            fill_particles(run, particles);
        }
    }
}

void HistogramRunner::fill(const Block& block) {
    std::visit(
        [this](const auto& b) {
            using T = std::decay_t<decltype(b)>;

            for (auto& run : runs_) {
                const auto& request = run.request;

                if constexpr (std::is_same_v<T, ParticleBlock>) {
                    if (request.target == HistogramTarget::Particles) {
                        fill_particles(run, b.particles);
                    }
                }

                if constexpr (std::is_same_v<T, EndBlock>) {
                    if (request.target == HistogramTarget::EndBlock) {
                        fill_block_histogram(run, b,
                                             request.end_block_quantities,
                                             request.end_block_weight,
                                             end_block_value);
                    }
                }

                if constexpr (std::is_same_v<T, InteractionBlock>) {
                    switch (request.target) {
                        case HistogramTarget::Interaction:
                            fill_block_histogram(
                                run, b, request.interaction_quantities,
                                request.interaction_weight, interaction_value);
                            break;

                        case HistogramTarget::Incoming:
                        case HistogramTarget::Outgoing: {
                            const double block_weight =
                                request.interaction_weight
                                    ? interaction_value(
                                          b, *request.interaction_weight)
                                    : 1.0;

                            fill_particles(
                                run,
                                request.target == HistogramTarget::Incoming
                                    ? b.incoming
                                    : b.outgoing,
                                block_weight);
                            break;
                        }

                        default:
                            break;
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
