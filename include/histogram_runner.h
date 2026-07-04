#ifndef BRASS_HISTOGRAM_RUNNER_H
#define BRASS_HISTOGRAM_RUNNER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
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

HistogramResult histogram_particles(
    const Particles& particles,
    const std::vector<std::string>& histogram_quantities,
    const std::vector<AxisConfig>& axes);

HistogramResult histogram_reader(
    BinaryReader& reader, const std::vector<std::string>& histogram_quantities,
    const std::vector<AxisConfig>& axes);

std::unordered_map<int32_t, HistogramResult> histograms_by_particles(
    const Particles& particles,
    const std::vector<std::string>& histogram_quantities,
    const std::vector<AxisConfig>& axes, const std::string& by,
    const std::vector<int32_t>& group_values);

std::unordered_map<int32_t, HistogramResult> histograms_by_reader(
    BinaryReader& reader, const std::vector<std::string>& histogram_quantities,
    const std::vector<AxisConfig>& axes, const std::string& by,
    const std::vector<int32_t>& group_values);

}  // namespace brass

#endif  // BRASS_HISTOGRAM_RUNNER_H
