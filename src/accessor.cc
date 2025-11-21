#include "accessor.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <utility>

using std::size_t;

int32_t Accessor::get_int(const std::string& name, const ParticleBlock& block,
                          size_t i) const {
    return quantity<int32_t>(name, block, i);
}

double Accessor::get_double(const std::string& name, const ParticleBlock& block,
                            size_t i) const {
    return quantity<double>(name, block, i);
}

Accessor::QuantityHandle Accessor::resolve(const std::string& name) const {
    if (!layout) {
        throw std::runtime_error("Accessor::resolve: layout not set");
    }

    auto it_off = layout->find(name);
    if (it_off == layout->end()) {
        throw std::runtime_error(
            "Accessor::resolve: unknown quantity in layout: " + name);
    }

    auto it_ty = quantity_string_map.find(name);
    if (it_ty == quantity_string_map.end()) {
        throw std::runtime_error(
            "Accessor::resolve: unknown quantity type for: " + name);
    }

    return QuantityHandle{it_off->second, it_ty->second};
}

void Accessor::set_resolved_fields(const std::vector<std::string>& names) {
    resolved_fields.clear();
    resolved_fields.reserve(names.size());
    for (const auto& n : names) {
        resolved_fields.push_back(ResolvedField{n, resolve(n)});
    }
}

std::vector<Accessor::ResolvedField> Accessor::make_resolved_fields(
    const std::vector<std::string>& names) const {
    std::vector<ResolvedField> out;
    out.reserve(names.size());
    for (const auto& n : names) {
        out.push_back(ResolvedField{n, resolve(n)});
    }
    return out;
}

py::list Accessor::gather_arrays_resolved(
    const char* base, size_t count, size_t stride,
    const std::vector<ResolvedField>& fields) const {
    py::list out;

    for (const auto& f : fields) {
        const char* p = base + f.h.offset;

        if (f.h.type == QuantityType::Double) {
            py::array_t<double> arr(count);
            auto a = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < count; ++i) {
                a(i) = *reinterpret_cast<const double*>(p);
                p += stride;
            }
            out.append(py::make_tuple(f.name, std::move(arr)));
        } else {
            // default to Int32 for everything else (QuantityType::Int32)
            py::array_t<int32_t> arr(count);
            auto a = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < count; ++i) {
                a(i) = *reinterpret_cast<const int32_t*>(p);
                p += stride;
            }
            out.append(py::make_tuple(f.name, std::move(arr)));
        }
    }

    return out;
}

py::list Accessor::gather_arrays_default(const char* base, size_t count,
                                         size_t stride) const {
    return gather_arrays_resolved(base, count, stride, resolved_fields);
}
