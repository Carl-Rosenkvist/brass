#ifndef BRASS_PARTICLES_H
#define BRASS_PARTICLES_H

#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "quantities.h"

namespace brass {

using Column = std::variant<std::vector<double>, std::vector<int32_t>>;

class Particles {
   public:
    Particles() = default;

    Particles(std::vector<std::byte> bytes,
              std::shared_ptr<const ParticleLayout> layout)
        : layout_(std::move(layout)) {
        make_columns(bytes);
    }

    std::size_t particle_size() const {
        return layout_ ? layout_->particle_size : 0;
    }

    std::size_t size() const { return size_; }

    bool empty() const { return size_ == 0; }

    const ParticleLayout& layout() const {
        if (!layout_) {
            throw std::runtime_error("Particles: no layout");
        }

        return *layout_;
    }

    const QuantityOffset& quantity(const std::string& name) const {
        return layout().quantity(name);
    }

    const Column& column_variant(const std::string& name) const {
        const auto it = columns_.find(name);

        if (it == columns_.end()) {
            throw std::runtime_error("unknown particle quantity: " + name);
        }

        return it->second;
    }

    template <class T>
    const std::vector<T>& column(const std::string& name) const {
        const auto& col = column_variant(name);

        const auto* ptr = std::get_if<std::vector<T>>(&col);
        if (!ptr) {
            throw std::runtime_error("wrong type for particle quantity: " +
                                     name);
        }

        return *ptr;
    }

   private:
    void make_columns(std::span<const std::byte> bytes) {
        const auto psize = particle_size();

        if (psize == 0) {
            size_ = 0;
            return;
        }

        if (bytes.size() % psize != 0) {
            throw std::runtime_error(
                "Particles: byte size is not divisible by particle size");
        }

        size_ = bytes.size() / psize;

        for (const auto& q : layout().offsets) {
            if (q.size == sizeof(double)) {
                std::vector<double> values(size_);

                for (std::size_t i = 0; i < size_; ++i) {
                    std::memcpy(&values[i], bytes.data() + i * psize + q.offset,
                                sizeof(double));
                }

                columns_.emplace(q.name, std::move(values));
                continue;
            }

            if (q.size == sizeof(int32_t)) {
                std::vector<int32_t> values(size_);

                for (std::size_t i = 0; i < size_; ++i) {
                    std::memcpy(&values[i], bytes.data() + i * psize + q.offset,
                                sizeof(int32_t));
                }

                columns_.emplace(q.name, std::move(values));
                continue;
            }

            throw std::runtime_error("unsupported quantity size: " + q.name);
        }
    }

    std::shared_ptr<const ParticleLayout> layout_;
    std::size_t size_ = 0;
    std::unordered_map<std::string, Column> columns_;
};

}  // namespace brass

#endif  // BRASS_PARTICLES_H
