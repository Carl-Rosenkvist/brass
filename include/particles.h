#ifndef BRASS_PARTICLES_H
#define BRASS_PARTICLES_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "quantities.h"

namespace brass {

using Column = std::variant<std::vector<double>, std::vector<int32_t>>;

class ParticleRow {
   public:
    explicit ParticleRow(const std::byte* row) : row_(row) {}

    template <typename T>
    T get_value(std::size_t offset) const {
        static_assert(
            std::is_trivially_copyable_v<T>,
            "ParticleRow::get_value<T> requires a trivially copyable type");

        T value{};
        std::memcpy(&value, row_ + offset, sizeof(T));
        return value;
    }

    double get_double(const QuantityOffset& q) const {
        if (q.size != sizeof(double)) {
            throw std::runtime_error("quantity is not double: " + q.name);
        }

        return get_value<double>(q.offset);
    }

    int32_t get_int32(const QuantityOffset& q) const {
        if (q.size != sizeof(int32_t)) {
            throw std::runtime_error("quantity is not int32: " + q.name);
        }

        return get_value<int32_t>(q.offset);
    }

   private:
    const std::byte* row_;
};

class Particles {
   public:
    Particles() = default;

    Particles(std::vector<std::byte> bytes,
              std::shared_ptr<const ParticleLayout> layout)
        : bytes_(std::move(bytes)), layout_(std::move(layout)) {
        const auto psize = particle_size();

        if (psize == 0) {
            size_ = 0;
            return;
        }

        if (bytes_.size() % psize != 0) {
            throw std::runtime_error(
                "Particles: byte size is not divisible by particle size");
        }

        size_ = bytes_.size() / psize;
    }

    std::size_t particle_size() const {
        return layout_ ? layout_->particle_size : 0;
    }

    std::size_t size() const { return size_; }

    bool empty() const { return size_ == 0; }

    const std::byte* data() const { return bytes_.data(); }

    std::size_t bytes_size() const { return bytes_.size(); }

    const ParticleLayout& layout() const {
        if (!layout_) {
            throw std::runtime_error("Particles: no layout");
        }

        return *layout_;
    }

    const QuantityOffset& quantity(const std::string& name) const {
        return layout().quantity(name);
    }

    ParticleRow row(std::size_t i) const {
        if (i >= size_) {
            throw std::runtime_error("particle row index out of range");
        }

        return ParticleRow(bytes_.data() + i * particle_size());
    }

    const Column& column_variant(const std::string& name) const {
        const auto it = columns_.find(name);

        if (it != columns_.end()) {
            return it->second;
        }

        make_column(name);

        const auto inserted = columns_.find(name);
        if (inserted == columns_.end()) {
            throw std::runtime_error("unknown particle quantity: " + name);
        }

        return inserted->second;
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
    void make_column(const std::string& name) const {
        const auto& q = quantity(name);
        const auto psize = particle_size();

        if (q.size == sizeof(double)) {
            std::vector<double> values(size_);

            for (std::size_t i = 0; i < size_; ++i) {
                std::memcpy(&values[i], bytes_.data() + i * psize + q.offset,
                            sizeof(double));
            }

            columns_.emplace(q.name, std::move(values));
            return;
        }

        if (q.size == sizeof(int32_t)) {
            std::vector<int32_t> values(size_);

            for (std::size_t i = 0; i < size_; ++i) {
                std::memcpy(&values[i], bytes_.data() + i * psize + q.offset,
                            sizeof(int32_t));
            }

            columns_.emplace(q.name, std::move(values));
            return;
        }

        throw std::runtime_error("unsupported quantity size: " + q.name);
    }

    std::vector<std::byte> bytes_;
    std::shared_ptr<const ParticleLayout> layout_;
    std::size_t size_ = 0;

    mutable std::unordered_map<std::string, Column> columns_;
};

}  // namespace brass

#endif  // BRASS_PARTICLES_H
