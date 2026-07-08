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

struct HistogramRun {
    HistogramRequest request;
    HistogramStorage histograms;
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

    template <class BlockType, class Value>
    void fill_block_histograms(std::vector<HistogramRun>& runs,
                               HistogramTarget target, const BlockType& block,
                               Value value) {
        for (auto& run : runs) {
            if (run.request.target != target) {
                continue;
            }

            if (run.request.group_by) {
                throw std::runtime_error(
                    "group_by is not supported for block histograms");
            }

            auto& hist = std::get<Histogram>(run.histograms);

            if (run.request.quantities.size() != hist.rank()) {
                throw std::runtime_error(
                    "histogram: quantity/axis dimension mismatch");
            }

            std::vector<double> values(run.request.quantities.size());

            for (std::size_t d = 0; d < run.request.quantities.size(); ++d) {
                values[d] = value(block, run.request.quantities[d]);
            }

            double weight = 1.0;

            if (run.request.weight_quantity) {
                weight = value(block, *run.request.weight_quantity);
            }

            hist.fill_values(values, weight);
        }
    }
};

}  // namespace brass

#endif  // BRASS_HISTOGRAM_RUNNER_H
