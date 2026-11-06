#include "particles.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace brass {

double ParticleRow::get_double(const QuantityOffset& q) const {
    if (q.type != QuantityType::Double) {
        throw std::runtime_error("quantity is not double: " + q.name);
    }

    return get_value<double>(q.offset);
}

int32_t ParticleRow::get_int32(const QuantityOffset& q) const {
    if (q.type != QuantityType::Int32) {
        throw std::runtime_error("quantity is not int32: " + q.name);
    }

    return get_value<int32_t>(q.offset);
}

double ParticleRow::get_as_double(const QuantityOffset& q) const {
    switch (q.type) {
        case QuantityType::Double:
            return get_value<double>(q.offset);
        case QuantityType::Int32:
            return static_cast<double>(get_value<int32_t>(q.offset));
    }

    throw std::logic_error("unhandled quantity type: " + q.name);
}

Particles::Particles(ByteBuffer bytes,
                     std::shared_ptr<const ParticleLayout> layout)
    : bytes_(std::move(bytes)), layout_(std::move(layout)) {
    const auto size = particle_size();

    if (size == 0) {
        size_ = 0;
        return;
    }

    if (bytes_.size() % size != 0) {
        throw std::runtime_error(
            "Particles: byte size is not divisible by particle size");
    }

    size_ = bytes_.size() / size;
}

std::size_t Particles::particle_size() const {
    return layout_ ? layout_->particle_size : 0;
}

const ParticleLayout& Particles::layout() const {
    if (!layout_) {
        throw std::runtime_error("Particles: no layout");
    }

    return *layout_;
}

const QuantityOffset& Particles::quantity(Quantity q) const {
    return layout().at(q);
}

const QuantityOffset* Particles::find_quantity(Quantity q) const {
    return layout().find(q);
}

ParticleRow Particles::row(std::size_t i) const {
    if (i >= size_) {
        throw std::runtime_error("particle row index out of range");
    }

    return ParticleRow(bytes_.data() + i * particle_size());
}

double Particles::value(const ParticleRow& row, Quantity q) const {
    const auto& layout = this->layout();

    if (const auto* stored = layout.find(q)) {
        return row.get_as_double(*stored);
    }

    const auto get = [&](Quantity stored) {
        return row.get_double(layout.at(stored));
    };

    switch (q) {
        case Quantity::pt:
            return std::hypot(get(Quantity::px), get(Quantity::py));

        case Quantity::p: {
            const auto px = get(Quantity::px);
            const auto py = get(Quantity::py);
            const auto pz = get(Quantity::pz);
            return std::sqrt(px * px + py * py + pz * pz);
        }

        case Quantity::mass: {
            const auto p0 = get(Quantity::p0);
            const auto px = get(Quantity::px);
            const auto py = get(Quantity::py);
            const auto pz = get(Quantity::pz);
            const auto m2 = p0 * p0 - px * px - py * py - pz * pz;
            return m2 > 0.0 ? std::sqrt(m2) : 0.0;
        }

        case Quantity::mt: {
            const auto m = value(row, Quantity::mass);
            const auto pt = value(row, Quantity::pt);
            return std::sqrt(m * m + pt * pt);
        }

        case Quantity::y_rap: {
            const auto p0 = get(Quantity::p0);
            const auto pz = get(Quantity::pz);
            return 0.5 * std::log((p0 + pz) / (p0 - pz));
        }

        default:
            // Not stored and not computable: reports "not in file".
            return row.get_as_double(layout.at(q));
    }
}

double Particles::value(std::size_t i, Quantity q) const {
    return value(row(i), q);
}

void Particles::compute(Quantity q, std::vector<double>& out) const {
    out.resize(size_);

    if (size_ == 0) {
        return;
    }

    const auto& layout = this->layout();
    const auto stride = particle_size();
    const std::byte* base = bytes_.data();

    const auto load = [&](std::size_t offset, std::size_t i) {
        double value;
        std::memcpy(&value, base + i * stride + offset, sizeof(double));
        return value;
    };

    if (const auto* stored = layout.find(q)) {
        const auto offset = stored->offset;

        if (stored->type == QuantityType::Double) {
            for (std::size_t i = 0; i < size_; ++i) {
                out[i] = load(offset, i);
            }
        } else {
            for (std::size_t i = 0; i < size_; ++i) {
                int32_t value;
                std::memcpy(&value, base + i * stride + offset, sizeof(value));
                out[i] = static_cast<double>(value);
            }
        }

        return;
    }

    // Offset of a stored double quantity needed for a computed one.
    const auto offset_of = [&](Quantity needed) {
        const auto& stored = layout.at(needed);

        if (stored.type != QuantityType::Double) {
            throw std::runtime_error("quantity is not double: " + stored.name);
        }

        return stored.offset;
    };

    switch (q) {
        case Quantity::pt: {
            const auto px = offset_of(Quantity::px);
            const auto py = offset_of(Quantity::py);
            for (std::size_t i = 0; i < size_; ++i) {
                out[i] = std::hypot(load(px, i), load(py, i));
            }
            return;
        }

        case Quantity::p: {
            const auto px = offset_of(Quantity::px);
            const auto py = offset_of(Quantity::py);
            const auto pz = offset_of(Quantity::pz);
            for (std::size_t i = 0; i < size_; ++i) {
                const auto x = load(px, i);
                const auto y = load(py, i);
                const auto z = load(pz, i);
                out[i] = std::sqrt(x * x + y * y + z * z);
            }
            return;
        }

        case Quantity::mass: {
            const auto p0 = offset_of(Quantity::p0);
            const auto px = offset_of(Quantity::px);
            const auto py = offset_of(Quantity::py);
            const auto pz = offset_of(Quantity::pz);
            for (std::size_t i = 0; i < size_; ++i) {
                const auto e = load(p0, i);
                const auto x = load(px, i);
                const auto y = load(py, i);
                const auto z = load(pz, i);
                const auto m2 = e * e - x * x - y * y - z * z;
                out[i] = m2 > 0.0 ? std::sqrt(m2) : 0.0;
            }
            return;
        }

        case Quantity::mt: {
            // mass is stored or computed; reuse `out` for it.
            compute(Quantity::mass, out);
            const auto px = offset_of(Quantity::px);
            const auto py = offset_of(Quantity::py);
            for (std::size_t i = 0; i < size_; ++i) {
                const auto m = out[i];
                const auto pt = std::hypot(load(px, i), load(py, i));
                out[i] = std::sqrt(m * m + pt * pt);
            }
            return;
        }

        case Quantity::y_rap: {
            const auto p0 = offset_of(Quantity::p0);
            const auto pz = offset_of(Quantity::pz);
            for (std::size_t i = 0; i < size_; ++i) {
                const auto e = load(p0, i);
                const auto z = load(pz, i);
                out[i] = 0.5 * std::log((e + z) / (e - z));
            }
            return;
        }

        default:
            // Not stored and not computable: reports "not in file".
            layout.at(q);
            return;
    }
}

const Column& Particles::column_variant(Quantity q) const {
    if (const auto it = columns_.find(q); it != columns_.end()) {
        return it->second;
    }

    make_column(quantity(q));

    return columns_.at(q);
}

void Particles::make_column(const QuantityOffset& q) const {
    switch (q.type) {
        case QuantityType::Double:
            make_typed_column<double>(q);
            return;
        case QuantityType::Int32:
            make_typed_column<int32_t>(q);
            return;
    }

    throw std::logic_error("unhandled quantity type: " + q.name);
}

}  // namespace brass
