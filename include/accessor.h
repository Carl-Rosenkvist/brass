#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "quantities.h"         // QuantityType, quantity_string_map, get_quantity(...)
#include "blocks.h"

class Accessor {
public:
    virtual ~Accessor() = default;

    virtual void on_particle_block(const ParticleBlock& block) {}
    virtual void on_interaction_block(const InteractionBlock& block) {}
    virtual void on_end_block(const EndBlock& block) {}
    virtual void on_header(Header& header_in) {}

    void set_layout(const std::unordered_map<std::string, size_t>* layout_in) {
        layout = layout_in;
    }

    const std::unordered_map<std::string, size_t>& layout_map() const {
        if (!layout) throw std::runtime_error("Layout not set in Accessor");
        return *layout;
    }

    template <typename T>
    T quantity(const std::string& name, const ParticleBlock& block,
               size_t particle_index) const {
        if (!layout) throw std::runtime_error("Layout not set in Accessor");
        if (particle_index >= block.npart)
            throw std::out_of_range("Invalid particle index");
        return get_quantity<T>(block.particle(particle_index), name, *layout);
    }

    int32_t get_int(const std::string& name, const ParticleBlock& block, size_t i) const;
    double  get_double(const std::string& name, const ParticleBlock& block, size_t i) const;

    // Resolve-once handle for hot loops
    struct QuantityHandle {
        size_t offset;
        QuantityType type;
    };

    struct ResolvedField {
        std::string name;
        QuantityHandle h;
    };

    // Resolve a single quantity name to offset+type
    QuantityHandle resolve(const std::string& name) const;

    // Build and store a default resolved field list
    void set_resolved_fields(const std::vector<std::string>& names);

    // Build and return a temporary resolved list (no storing)
    std::vector<ResolvedField> make_resolved_fields(const std::vector<std::string>& names) const;

    // Core gather function: use any resolved fields vector
    py::list gather_arrays_resolved(const char* base, size_t count, size_t stride,
                                    const std::vector<ResolvedField>& fields) const;

    // Use the pre-stored resolved fields
    py::list gather_arrays_default(const char* base, size_t count, size_t stride) const;

    // --- Low-level helpers (kept inline/hot) ---
    inline double get_double_fast(const ParticleBlock& b, size_t off, size_t i) const noexcept {
        const char* p = b.particles.data() + i * b.particle_size + off;
        return *reinterpret_cast<const double*>(p);
    }

    inline int32_t get_int_fast(const ParticleBlock& b, size_t off, size_t i) const noexcept {
        const char* p = b.particles.data() + i * b.particle_size + off;
        return *reinterpret_cast<const int32_t*>(p);
    }

    inline double get_double_fast(const ParticleBlock& b, const QuantityHandle& h, size_t i) const noexcept {
#ifndef NDEBUG
        if (h.type != QuantityType::Double)
            throw std::logic_error("get_double_fast: wrong type");
#endif
        return get_double_fast(b, h.offset, i);
    }

    inline int32_t get_int_fast(const ParticleBlock& b, const QuantityHandle& h, size_t i) const noexcept {
#ifndef NDEBUG
        if (h.type != QuantityType::Int32)
            throw std::logic_error("get_int_fast: wrong type");
#endif
        return get_int_fast(b, h.offset, i);
    }

protected:
    const std::unordered_map<std::string, size_t>* layout = nullptr;
    std::optional<Header> header = std::nullopt;
    std::vector<ResolvedField> resolved_fields;
};
