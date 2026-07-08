#include "dilepton_histogrammer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace brass {

struct DileptonFourVector {
    double p0 = 0.0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
};

static HistogramResult make_histogram_result(const Histogram& hist) {
    return HistogramResult{
        .values = hist.values(),
        .edges = hist.edges(),
        .shape = hist.shape(),
    };
}

static std::optional<DileptonFourVector> outgoing_pair(
    const InteractionBlock& block, std::pair<int32_t, int32_t> pair_pdgs) {
    DileptonFourVector pair;

    std::vector<int32_t> found_pdgs;
    found_pdgs.reserve(2);

    for (std::size_t i = 0; i < block.outgoing.size(); ++i) {
        const auto row = block.outgoing.row(i);
        const auto pdg = static_cast<int32_t>(block.outgoing.value(row, "pdg"));

        if (pdg != pair_pdgs.first && pdg != pair_pdgs.second) {
            continue;
        }

        found_pdgs.push_back(pdg);

        pair.p0 += block.outgoing.value(row, "p0");
        pair.px += block.outgoing.value(row, "px");
        pair.py += block.outgoing.value(row, "py");
        pair.pz += block.outgoing.value(row, "pz");
    }

    std::vector<int32_t> expected_pdgs{
        pair_pdgs.first,
        pair_pdgs.second,
    };

    std::sort(found_pdgs.begin(), found_pdgs.end());
    std::sort(expected_pdgs.begin(), expected_pdgs.end());

    if (found_pdgs != expected_pdgs) {
        return std::nullopt;
    }

    return pair;
}

static double dilepton_mass(const DileptonFourVector& p) {
    const double m2 = p.p0 * p.p0 - p.px * p.px - p.py * p.py - p.pz * p.pz;

    return std::sqrt(std::max(m2, 0.0));
}

static double dilepton_momentum(const DileptonFourVector& p) {
    return std::sqrt(p.px * p.px + p.py * p.py + p.pz * p.pz);
}

static double dilepton_pt(const DileptonFourVector& p) {
    return std::hypot(p.px, p.py);
}

static double dilepton_rapidity(const DileptonFourVector& p) {
    return 0.5 * std::log((p.p0 + p.pz) / (p.p0 - p.pz));
}

static double dilepton_value(const DileptonFourVector& p,
                             const std::string& name) {
    if (name == "p0") {
        return p.p0;
    }

    if (name == "px") {
        return p.px;
    }

    if (name == "py") {
        return p.py;
    }

    if (name == "pz") {
        return p.pz;
    }

    if (name == "p") {
        return dilepton_momentum(p);
    }

    if (name == "pt") {
        return dilepton_pt(p);
    }

    if (name == "mass") {
        return dilepton_mass(p);
    }

    if (name == "mt") {
        return std::hypot(dilepton_mass(p), dilepton_pt(p));
    }

    if (name == "y_rap") {
        return dilepton_rapidity(p);
    }

    throw std::runtime_error("unknown dilepton quantity: " + name);
}

static std::string dilepton_channel_key(const InteractionBlock& block) {
    if (block.incoming.size() != 1) {
        return {};
    }

    const auto parent_row = block.incoming.row(0);
    const auto parent_pdg =
        static_cast<int32_t>(block.incoming.value(parent_row, "pdg"));

    std::vector<int32_t> daughters;
    daughters.reserve(block.outgoing.size());

    for (std::size_t i = 0; i < block.outgoing.size(); ++i) {
        const auto row = block.outgoing.row(i);

        daughters.push_back(
            static_cast<int32_t>(block.outgoing.value(row, "pdg")));
    }

    std::sort(daughters.begin(), daughters.end());

    std::string key = std::to_string(parent_pdg) + " ->";

    for (const auto pdg : daughters) {
        key += " " + std::to_string(pdg);
    }

    return key;
}

DileptonHistogrammer::DileptonHistogrammer(
    std::vector<std::string> quantities, std::vector<AxisConfig> axes,
    std::pair<int32_t, int32_t> pair_pdgs, std::string weight_quantity)
    : quantities_(std::move(quantities)),
      axes_(std::move(axes)),
      pair_pdgs_(pair_pdgs),
      weight_quantity_(std::move(weight_quantity)) {}

void DileptonHistogrammer::fill(const InteractionBlock& block) {
    const auto pair = outgoing_pair(block, pair_pdgs_);

    if (!pair) {
        return;
    }

    const auto key = dilepton_channel_key(block);

    if (key.empty()) {
        return;
    }

    auto [it, inserted] = histograms_.try_emplace(key, Histogram(axes_));

    auto& hist = it->second;

    if (quantities_.size() != hist.rank()) {
        throw std::runtime_error(
            "DileptonHistogrammer: quantity/axis dimension mismatch");
    }

    std::vector<double> values(quantities_.size());

    for (std::size_t i = 0; i < quantities_.size(); ++i) {
        values[i] = dilepton_value(*pair, quantities_[i]);
    }

    double weight = 1.0;

    if (!weight_quantity_.empty()) {
        weight = interaction_value(block, weight_quantity_);
    }

    hist.fill_values(values, weight);
}

void DileptonHistogrammer::fill(const Block& block) {
    std::visit(
        [this](const auto& b) {
            using T = std::decay_t<decltype(b)>;

            if constexpr (std::is_same_v<T, InteractionBlock>) {
                fill(b);
            }
        },
        block);
}

void DileptonHistogrammer::fill(BinaryReader& reader) {
    while (true) {
        auto block = reader.read();

        if (!block) {
            break;
        }

        fill(*block);
    }
}

std::map<std::string, HistogramResult> DileptonHistogrammer::result() const {
    std::map<std::string, HistogramResult> results;

    for (const auto& [key, hist] : histograms_) {
        results.emplace(key, make_histogram_result(hist));
    }

    return results;
}

}  // namespace brass
