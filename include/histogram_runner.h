#ifndef BRASS_HISTOGRAM_RUNNER_H
#define BRASS_HISTOGRAM_RUNNER_H

#include <cstddef>
#include <cstdint>
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

struct HistogramResult {
    std::vector<double> values;
    std::vector<std::vector<double>> edges;
    std::vector<std::size_t> shape;

    HistogramResult& operator+=(const HistogramResult& other) {
        if (shape != other.shape || edges != other.edges) {
            throw std::invalid_argument(
                "cannot add histograms with different binning");
        }

        if (values.size() != other.values.size()) {
            throw std::invalid_argument(
                "cannot add histograms with different sizes");
        }

        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] += other.values[i];
        }

        return *this;
    }
};
using GroupedHistogramResult = std::unordered_map<int32_t, HistogramResult>;

using HistogramRunResult =
    std::variant<HistogramResult, GroupedHistogramResult>;

using HistogramBatchResult = std::vector<HistogramRunResult>;

struct HistogramGroupBy {
    std::string quantity;
    std::vector<int32_t> values;
};

enum class HistogramTarget {
    Particles,
    Incoming,
    Outgoing,
    Header,
    EndBlock,
    Interaction,
};

struct HistogramRequest {
    std::vector<std::string> quantities;
    std::vector<AxisConfig> axes;
    std::optional<HistogramGroupBy> group_by;
    std::optional<std::string> weight_quantity;
    HistogramTarget target = HistogramTarget::Particles;
};

using HistogramStorage =
    std::variant<Histogram, std::unordered_map<int32_t, Histogram>>;

// A HistogramRequest with all names resolved to enums. Built once when the
// runner is constructed, so no names are looked up while filling.
struct ResolvedRequest {
    HistogramTarget target = HistogramTarget::Particles;

    // Axis quantities; which vector is used depends on the target.
    std::vector<Quantity> particle_quantities;  // Particles/Incoming/Outgoing
    std::vector<InteractionQuantity> interaction_quantities;  // Interaction
    std::vector<EndBlockQuantity> end_block_quantities;       // EndBlock
    std::vector<HeaderQuantity> header_quantities;            // Header

    std::optional<Quantity> group_by;

    // At most one of these is set.
    std::optional<Quantity> particle_weight;
    std::optional<InteractionQuantity> interaction_weight;
    std::optional<EndBlockQuantity> end_block_weight;
    std::optional<HeaderQuantity> header_weight;
};

ResolvedRequest resolve_request(const HistogramRequest& request);

// Values collected from one block before they are filled in one call.
// Reused across blocks so that filling does not allocate per block.
struct FillBuffer {
    std::vector<std::vector<double>> columns;  // one per axis
    std::vector<double> weights;               // empty when unweighted

    void reset(std::size_t rank) {
        columns.resize(rank);
        for (auto& column : columns) {
            column.clear();
        }
        weights.clear();
    }

    std::size_t size() const {
        return columns.empty() ? 0 : columns.front().size();
    }
};

struct HistogramRun {
    ResolvedRequest request;
    HistogramStorage histograms;

    FillBuffer buffer;  // whole block, all particles
    std::unordered_map<int32_t, FillBuffer> group_buffers;  // grouped
};

class HistogramRunner {
   public:
    HistogramRunner(BinaryReader& reader,
                    std::vector<HistogramRequest> requests);

    void fill(const Particles& particles);
    void fill(const Block& block);
    void run();

    HistogramBatchResult result() const;

   private:
    BinaryReader& reader_;
    std::vector<HistogramRun> runs_;

    void fill_particles(HistogramRun& run, const Particles& particles,
                        double block_weight = 1.0);
};

}  // namespace brass

#endif  // BRASS_HISTOGRAM_RUNNER_H
