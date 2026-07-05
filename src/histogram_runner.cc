#include "histogram_runner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "binaryreader.h"
#include "histogram.h"
#include "particles.h"

namespace brass {

namespace {

enum class ParticleQuantityKind {
    Column,
    Pt,
    P,
    Mass,
    Mt,
    Rapidity,
};

ParticleQuantityKind quantity_kind_from_name(const std::string& name) {
    if (name == "pt") {
        return ParticleQuantityKind::Pt;
    }

    if (name == "p") {
        return ParticleQuantityKind::P;
    }

    if (name == "m" || name == "mass_from_momentum") {
        return ParticleQuantityKind::Mass;
    }

    if (name == "mt") {
        return ParticleQuantityKind::Mt;
    }

    if (name == "y" || name == "y_rap" || name == "rapidity") {
        return ParticleQuantityKind::Rapidity;
    }

    return ParticleQuantityKind::Column;
}

class ParticleQuantity {
   public:
    ParticleQuantity(const ParticleLayout& layout, std::string name)
        : name_(std::move(name)), kind_(quantity_kind_from_name(name_)) {
        switch (kind_) {
            case ParticleQuantityKind::Column:
                inputs_.push_back(layout.quantity(name_));
                break;

            case ParticleQuantityKind::Pt:
                inputs_.push_back(layout.quantity("px"));
                inputs_.push_back(layout.quantity("py"));
                break;

            case ParticleQuantityKind::P:
                inputs_.push_back(layout.quantity("px"));
                inputs_.push_back(layout.quantity("py"));
                inputs_.push_back(layout.quantity("pz"));
                break;

            case ParticleQuantityKind::Mass:
            case ParticleQuantityKind::Mt:
                inputs_.push_back(layout.quantity("p0"));
                inputs_.push_back(layout.quantity("px"));
                inputs_.push_back(layout.quantity("py"));
                inputs_.push_back(layout.quantity("pz"));
                break;

            case ParticleQuantityKind::Rapidity:
                inputs_.push_back(layout.quantity("p0"));
                inputs_.push_back(layout.quantity("pz"));
                break;
        }

        validate_inputs();
    }

    double value(const ParticleRow& row) const {
        switch (kind_) {
            case ParticleQuantityKind::Column:
                return row.get_double(inputs_[0]);

            case ParticleQuantityKind::Pt: {
                const double px = row.get_double(inputs_[0]);
                const double py = row.get_double(inputs_[1]);
                return std::hypot(px, py);
            }

            case ParticleQuantityKind::P: {
                const double px = row.get_double(inputs_[0]);
                const double py = row.get_double(inputs_[1]);
                const double pz = row.get_double(inputs_[2]);
                return std::hypot(std::hypot(px, py), pz);
            }

            case ParticleQuantityKind::Mass: {
                const double p0 = row.get_double(inputs_[0]);
                const double px = row.get_double(inputs_[1]);
                const double py = row.get_double(inputs_[2]);
                const double pz = row.get_double(inputs_[3]);

                const double m2 = p0 * p0 - px * px - py * py - pz * pz;
                return std::sqrt(std::max(m2, 0.0));
            }

            case ParticleQuantityKind::Mt: {
                const double p0 = row.get_double(inputs_[0]);
                const double px = row.get_double(inputs_[1]);
                const double py = row.get_double(inputs_[2]);
                const double pz = row.get_double(inputs_[3]);

                const double pt = std::hypot(px, py);
                const double m2 = p0 * p0 - px * px - py * py - pz * pz;
                const double mass = std::sqrt(std::max(m2, 0.0));

                return std::hypot(pt, mass);
            }

            case ParticleQuantityKind::Rapidity: {
                const double p0 = row.get_double(inputs_[0]);
                const double pz = row.get_double(inputs_[1]);

                if (p0 <= std::abs(pz)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }

                return 0.5 * std::log((p0 + pz) / (p0 - pz));
            }
        }

        throw std::runtime_error("unknown particle quantity: " + name_);
    }

   private:
    void validate_inputs() const {
        if (kind_ == ParticleQuantityKind::Column) {
            return;
        }

        for (const auto& q : inputs_) {
            if (q.size != sizeof(double)) {
                throw std::runtime_error("quantity is not double: " + q.name);
            }
        }
    }

    std::string name_;
    ParticleQuantityKind kind_ = ParticleQuantityKind::Column;
    std::vector<QuantityOffset> inputs_;
};

using ParticleQuantities = std::vector<ParticleQuantity>;

void check_histogram_request(const HistogramRequest& request,
                             const std::string& caller) {
    if (request.quantities.empty()) {
        throw std::runtime_error(caller + ": no histogram quantities given");
    }

    if (request.quantities.size() != request.axes.size()) {
        throw std::runtime_error(
            caller +
            ": number of histogram quantities must match number of axes");
    }
}

ParticleQuantities make_particle_quantities(
    const ParticleLayout& layout, const std::vector<std::string>& names) {
    ParticleQuantities quantities;
    quantities.reserve(names.size());

    for (const auto& name : names) {
        quantities.emplace_back(layout, name);
    }

    return quantities;
}

bool evaluate_particle_values(const ParticleRow& row,
                              const ParticleQuantities& quantities,
                              std::vector<double>& values) {
    for (std::size_t d = 0; d < quantities.size(); ++d) {
        values[d] = quantities[d].value(row);

        if (!std::isfinite(values[d])) {
            return false;
        }
    }

    return true;
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

HistogramResult make_result(const Histogram& hist) {
    return HistogramResult{
        .values = hist.values(),
        .edges = hist.edges(),
        .shape = hist.shape(),
    };
}

GroupedHistogramResult make_grouped_result(
    const std::unordered_map<int32_t, Histogram>& histograms) {
    GroupedHistogramResult out;
    out.reserve(histograms.size());

    for (const auto& [key, hist] : histograms) {
        out.emplace(key, make_result(hist));
    }

    return out;
}

struct HistogramRunState {
    HistogramRequest request;
    ParticleQuantities quantities;
    std::optional<QuantityOffset> group_quantity;
    std::variant<Histogram, std::unordered_map<int32_t, Histogram>> histograms;
};

HistogramRunState make_histogram_state(const ParticleLayout& layout,
                                       const HistogramRequest& request) {
    check_histogram_request(request, "histogram");

    HistogramRunState state{
        .request = request,
        .quantities = make_particle_quantities(layout, request.quantities),
        .group_quantity = std::nullopt,
        .histograms = Histogram(request.axes),
    };

    if (request.group_by.has_value()) {
        const auto& group_quantity =
            layout.quantity(request.group_by->quantity);

        if (group_quantity.size != sizeof(int32_t)) {
            throw std::runtime_error("quantity is not int32: " +
                                     request.group_by->quantity);
        }

        state.group_quantity = group_quantity;
        state.histograms =
            make_grouped_histograms(request.axes, request.group_by->values);
    }

    return state;
}

void fill_state_from_row(HistogramRunState& state, const ParticleRow& row,
                         std::vector<double>& values) {
    values.resize(state.quantities.size());

    if (!evaluate_particle_values(row, state.quantities, values)) {
        return;
    }

    if (state.request.group_by.has_value()) {
        const auto group = row.get_int32(*state.group_quantity);

        auto& histograms =
            std::get<std::unordered_map<int32_t, Histogram>>(state.histograms);

        const auto it = histograms.find(group);
        if (it == histograms.end()) {
            return;
        }

        it->second.fill_values(values);
        return;
    }

    auto& hist = std::get<Histogram>(state.histograms);
    hist.fill_values(values);
}

void fill_states_from_particles(std::vector<HistogramRunState>& states,
                                const Particles& particles) {
    std::vector<std::vector<double>> values(states.size());

    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto row = particles.row(i);

        for (std::size_t s = 0; s < states.size(); ++s) {
            fill_state_from_row(states[s], row, values[s]);
        }
    }
}

HistogramRunResult make_result(const HistogramRunState& state) {
    if (std::holds_alternative<Histogram>(state.histograms)) {
        return make_result(std::get<Histogram>(state.histograms));
    }

    return make_grouped_result(
        std::get<std::unordered_map<int32_t, Histogram>>(state.histograms));
}

std::vector<HistogramRunState> make_histogram_states(
    const ParticleLayout& layout,
    const std::vector<HistogramRequest>& requests) {
    std::vector<HistogramRunState> states;
    states.reserve(requests.size());

    for (const auto& request : requests) {
        states.push_back(make_histogram_state(layout, request));
    }

    return states;
}

HistogramBatchResult make_batch_result(
    const std::vector<HistogramRunState>& states) {
    HistogramBatchResult results;
    results.reserve(states.size());

    for (const auto& state : states) {
        results.push_back(make_result(state));
    }

    return results;
}

}  // namespace

HistogramRunResult histogram(const Particles& particles,
                             const HistogramRequest& request) {
    return histograms(particles, std::vector<HistogramRequest>{request}).at(0);
}

HistogramRunResult histogram(BinaryReader& reader,
                             const HistogramRequest& request) {
    return histograms(reader, std::vector<HistogramRequest>{request}).at(0);
}

HistogramBatchResult histograms(const Particles& particles,
                                const std::vector<HistogramRequest>& requests) {
    if (requests.empty()) {
        return {};
    }

    auto states = make_histogram_states(particles.layout(), requests);
    fill_states_from_particles(states, particles);

    return make_batch_result(states);
}

HistogramBatchResult histograms(BinaryReader& reader,
                                const std::vector<HistogramRequest>& requests) {
    if (requests.empty()) {
        return {};
    }

    std::optional<std::vector<HistogramRunState>> states;

    while (true) {
        auto block = reader.read();

        if (!block.has_value()) {
            break;
        }

        if (!std::holds_alternative<ParticleBlock>(*block)) {
            continue;
        }

        const auto& particles = std::get<ParticleBlock>(*block).particles;

        if (!states.has_value()) {
            states = make_histogram_states(particles.layout(), requests);
        }

        fill_states_from_particles(*states, particles);
    }

    if (!states.has_value()) {
        return {};
    }

    return make_batch_result(*states);
}

}  // namespace brass
