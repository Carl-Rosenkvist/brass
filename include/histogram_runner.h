#ifndef BRASS_HISTOGRAM_RUNNER_H
#define BRASS_HISTOGRAM_RUNNER_H

#include <cstddef>
#include <cstdint>
#include <optional>
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

struct HistogramRequest {
    std::vector<std::string> quantities;
    std::vector<AxisConfig> axes;
    std::optional<HistogramGroupBy> group_by;
};

HistogramRunResult histogram(const Particles& particles,
                             const HistogramRequest& request);

HistogramRunResult histogram(BinaryReader& reader,
                             const HistogramRequest& request);

HistogramBatchResult histograms(const Particles& particles,
                                const std::vector<HistogramRequest>& requests);

HistogramBatchResult histograms(BinaryReader& reader,
                                const std::vector<HistogramRequest>& requests);

}  // namespace brass

#endif  // BRASS_HISTOGRAM_RUNNER_H
