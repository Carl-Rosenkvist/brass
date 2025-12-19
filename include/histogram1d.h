#ifndef HISTOGRAM1D_H
#define HISTOGRAM1D_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

class Histogram1D {
   public:
    Histogram1D(double min, double max, size_t bins)
        : min_(min), max_(max), bins_(bins), counts_(bins, 0.0) {
        if (max <= min || bins == 0) {
            throw std::invalid_argument(
                "Invalid histogram range or bin count.");
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

    double bin_edge(size_t i) const {
        if (i > bins_) throw std::out_of_range("Invalid bin edge index");
        return min_ + i * bin_width_;
    }

    size_t num_bins() const { return bins_; }
    double bin_width() const { return bin_width_; }

    double bin_content(size_t i) const {
        if (i >= bins_) throw std::out_of_range("Invalid bin index");
        return counts_[i];
    }

    const std::vector<double>& counts() const { return counts_; }

    void scale(double factor) {
        for (double& c : counts_) c *= factor;
    }

    Histogram1D& operator+=(const Histogram1D& other) {
        if (bins_ != other.bins_ || min_ != other.min_ || max_ != other.max_) {
            throw std::runtime_error(
                "Cannot add histograms with different binning.");
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

inline bool operator==(const Histogram1D& a, const Histogram1D& b) {
    if (a.num_bins() != b.num_bins()) return false;
    for (size_t i = 0; i < a.num_bins(); ++i) {
        if (a.bin_content(i) != b.bin_content(i)) return false;
    }
    return true;
}

inline bool operator!=(const Histogram1D& a, const Histogram1D& b) {
    return !(a == b);
}

#include <pybind11/numpy.h>

inline pybind11::dict hist1d_to_state_dict(const Histogram1D& h) {
    pybind11::dict d;

    const size_t nbins = h.num_bins();

    // ---- edges array ----
    pybind11::array_t<double> edges(nbins + 1);
    auto edges_buf = edges.mutable_unchecked<1>();

    for (size_t i = 0; i <= nbins; ++i) {
        edges_buf(i) = h.bin_edge(i);
    }

    // ---- counts array ----
    const auto& counts_vec = h.counts();
    pybind11::array_t<double> counts(counts_vec.size());
    auto counts_buf = counts.mutable_unchecked<1>();

    for (size_t i = 0; i < counts_vec.size(); ++i) {
        counts_buf(i) = counts_vec[i];
    }

    d["counts"] = counts;

    return d;
}

#endif  // HISTOGRAM1D_H
