#ifndef HISTOGRAM1D_H
#define HISTOGRAM1D_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <yaml-cpp/yaml.h>
class Histogram1D {
public:
    Histogram1D(double min, double max, size_t bins)
        : min_(min), max_(max), bins_(bins), counts_(bins, 0.0) 
    {
        if (max <= min || bins == 0) {
            throw std::invalid_argument("Invalid histogram range or bin count.");
        }
        bin_width_ = (max - min) / bins;
    }

    
bool fill(double value, double weight = 1.0) {
    if (value < min_ || value >= max_) return false;
    size_t bin = static_cast<size_t>((value - min_) / bin_width_);
    counts_[bin] += weight;
    return true;
}
    double bin_center(size_t i) const {
        if (i >= bins_) throw std::out_of_range("Invalid bin index");
        return min_ + (i + 0.5) * bin_width_;
    }
    
    double get_bin_count(size_t i) const {
        if (i >= bins_) throw std::out_of_range("Invalid bin index");
        return counts_[i];
    }

double bin_edge(size_t i) const {
    if (i > bins_) throw std::out_of_range("Invalid bin edge index");
    return min_ + i * bin_width_;
}
    size_t num_bins() const { return bins_; }

    void print(std::ostream& out = std::cout) const {
        out << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < bins_; ++i) {
            out << bin_center(i) << "\t" << counts_[i] << "\n";
        }
    }

double bin_width() const { return bin_width_; }

void scale(double factor) {
    for (double& count : counts_) {
        count *= factor;
    }
}

double raw_bin_content(size_t i) const {
    return get_bin_count(i); // just for naming consistency
}

double bin_content(size_t i) const {
    return get_bin_count(i); // you could add smoothing, etc., later if needed
}
  
    Histogram1D& operator+=(const Histogram1D& other) {
        if (bins_ != other.bins_ || min_ != other.min_ || max_ != other.max_) {
            throw std::runtime_error("Cannot add histograms with different binning.");
        }
        for (size_t i = 0; i < bins_; ++i) {
            counts_[i] += other.counts_[i];
        }
            return *this;
    }



private:
    double min_, max_, bin_width_;
    size_t bins_;
    std::vector<double> counts_;
};

inline bool operator==(const Histogram1D& lhs, const Histogram1D& rhs) {
    return false;
}

inline bool operator!=(const Histogram1D& lhs, const Histogram1D& rhs) {
    return !(lhs == rhs);
}



inline void to_yaml(YAML::Emitter& out, const Histogram1D& h)
{
    out << YAML::BeginMap;

    out << YAML::Key << "values" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (size_t i = 0; i < h.num_bins(); ++i) {
        out << h.raw_bin_content(i);
    }
    out << YAML::EndSeq;

    out << YAML::Key << "centers" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (size_t i = 0; i < h.num_bins(); ++i) {
        out << h.bin_center(i);
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;
}

#endif // HISTOGRAM1D_H
