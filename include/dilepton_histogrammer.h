#ifndef BRASS_DILEPTON_HISTOGRAMMER_H
#define BRASS_DILEPTON_HISTOGRAMMER_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "binaryreader.h"
#include "histogram.h"
#include "histogram_runner.h"

namespace brass {

class DileptonHistogrammer {
   public:
    DileptonHistogrammer(std::vector<std::string> quantities,
                         std::vector<AxisConfig> axes,
                         std::pair<int32_t, int32_t> pair_pdgs =
                             std::pair<int32_t, int32_t>{-11, 11},
                         std::string weight_quantity = "weight");

    void fill(const InteractionBlock& block);
    void fill(const Block& block);
    void fill(BinaryReader& reader);

    std::map<std::string, HistogramResult> result() const;

   private:
    std::vector<std::string> quantities_;
    std::vector<AxisConfig> axes_;
    std::pair<int32_t, int32_t> pair_pdgs_;
    std::string weight_quantity_;

    std::map<std::string, Histogram> histograms_;
};

}  // namespace brass

#endif  // BRASS_DILEPTON_HISTOGRAMMER_H
