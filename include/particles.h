#ifndef BRASS_PARTICLES_H
#define BRASS_PARTICLES_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
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

    double get_double(const QuantityOffset& q) const;
    int32_t get_int32(const QuantityOffset& q) const;
    double get_as_double(const QuantityOffset& q) const;

   private:
    const std::byte* row_;
};

class Particles {
   public:
    Particles() = default;

    Particles(std::vector<std::byte> bytes,
              std::shared_ptr<const ParticleLayout> layout);

    std::size_t particle_size() const;
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    const std::byte* data() const { return bytes_.data(); }
    std::size_t bytes_size() const { return bytes_.size(); }

    const ParticleLayout& layout() const;

    const QuantityOffset& quantity(const std::string& name) const;
    const QuantityOffset* find_quantity(const std::string& name) const;

    ParticleRow row(std::size_t i) const;

    double stored_value(const ParticleRow& row, const std::string& name) const;

    double value(const ParticleRow& row, const std::string& name) const;

    double value(std::size_t i, const std::string& name) const;

    const Column& column_variant(const std::string& name) const;

    template <class T>
    const std::vector<T>& column(const std::string& name) const {
        const auto& col = column_variant(name);

        const auto* ptr = std::get_if<std::vector<T>>(&col);
        if (ptr == nullptr) {
            throw std::runtime_error("wrong type for particle quantity: " +
                                     name);
        }

        return *ptr;
    }

   private:
    void make_column(const std::string& name) const;

    template <class T>
    void make_column(const QuantityOffset& q) const {
        std::vector<T> values(size_);
        const auto size = particle_size();

        for (std::size_t i = 0; i < size_; ++i) {
            std::memcpy(&values[i], bytes_.data() + i * size + q.offset,
                        sizeof(T));
        }

        columns_.emplace(q.name, std::move(values));
    }

    std::vector<std::byte> bytes_;
    std::shared_ptr<const ParticleLayout> layout_;
    std::size_t size_ = 0;

    mutable std::unordered_map<std::string, Column> columns_;
};

}  // namespace brass

#endif  // BRASS_PARTICLES_H
