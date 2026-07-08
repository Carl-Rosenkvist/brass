#include "particles.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace brass {

double ParticleRow::get_double(const QuantityOffset& q) const {
    if (q.size != sizeof(double)) {
        throw std::runtime_error("quantity is not double: " + q.name);
    }

    return get_value<double>(q.offset);
}

int32_t ParticleRow::get_int32(const QuantityOffset& q) const {
    if (q.size != sizeof(int32_t)) {
        throw std::runtime_error("quantity is not int32: " + q.name);
    }

    return get_value<int32_t>(q.offset);
}

double ParticleRow::get_as_double(const QuantityOffset& q) const {
    if (q.size == sizeof(double)) {
        return get_double(q);
    }

    if (q.size == sizeof(int32_t)) {
        return static_cast<double>(get_int32(q));
    }

    throw std::runtime_error("unsupported quantity size: " + q.name);
}

Particles::Particles(std::vector<std::byte> bytes,
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

const QuantityOffset& Particles::quantity(const std::string& name) const {
    return layout().quantity(name);
}

const QuantityOffset* Particles::find_quantity(const std::string& name) const {
    const auto& offsets = layout().offsets;

    const auto it =
        std::find_if(offsets.begin(), offsets.end(),
                     [&](const QuantityOffset& q) { return q.name == name; });

    return it == offsets.end() ? nullptr : &*it;
}

ParticleRow Particles::row(std::size_t i) const {
    if (i >= size_) {
        throw std::runtime_error("particle row index out of range");
    }

    return ParticleRow(bytes_.data() + i * particle_size());
}

double Particles::stored_value(const ParticleRow& row,
                               const std::string& name) const {
    return row.get_as_double(quantity(name));
}

double Particles::value(const ParticleRow& row, const std::string& name) const {
    if (name == "pt") {
        const auto px = stored_value(row, "px");
        const auto py = stored_value(row, "py");
        return std::hypot(px, py);
    }

    if (name == "p") {
        const auto px = stored_value(row, "px");
        const auto py = stored_value(row, "py");
        const auto pz = stored_value(row, "pz");
        return std::sqrt(px * px + py * py + pz * pz);
    }

    if (name == "mass") {
        if (find_quantity("mass") != nullptr) {
            return stored_value(row, "mass");
        }

        const auto p0 = stored_value(row, "p0");
        const auto px = stored_value(row, "px");
        const auto py = stored_value(row, "py");
        const auto pz = stored_value(row, "pz");

        const auto m2 = p0 * p0 - px * px - py * py - pz * pz;
        return m2 > 0.0 ? std::sqrt(m2) : 0.0;
    }

    if (name == "mt") {
        const auto m = value(row, "mass");
        const auto pt = value(row, "pt");
        return std::sqrt(m * m + pt * pt);
    }
    if (name == "rapidity" || name == "y_rap") {
        const auto p0 = stored_value(row, "p0");
        const auto pz = stored_value(row, "pz");
        return 0.5 * std::log((p0 + pz) / (p0 - pz));
    }

    return stored_value(row, name);
}

double Particles::value(std::size_t i, const std::string& name) const {
    return value(row(i), name);
}

const Column& Particles::column_variant(const std::string& name) const {
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

void Particles::make_column(const std::string& name) const {
    const auto& q = quantity(name);

    if (q.size == sizeof(double)) {
        make_column<double>(q);
        return;
    }

    if (q.size == sizeof(int32_t)) {
        make_column<int32_t>(q);
        return;
    }

    throw std::runtime_error("unsupported quantity size: " + q.name);
}

}  // namespace brass
