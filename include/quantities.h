#ifndef BRASS_QUANTITIES_H
#define BRASS_QUANTITIES_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace brass {

struct QuantityOffset {
    std::string name;
    std::size_t offset = 0;
    std::size_t size = 0;
};

struct ParticleLayout {
    std::vector<QuantityOffset> offsets;
    std::size_t particle_size = 0;

    const QuantityOffset& quantity(const std::string& name) const {
        const auto it = std::find_if(
            offsets.begin(), offsets.end(),
            [&](const QuantityOffset& q) { return q.name == name; });

        if (it == offsets.end()) {
            throw std::runtime_error("unknown particle quantity: " + name);
        }

        return *it;
    }
};

inline const std::unordered_map<std::string, std::size_t> quantity_sizes = {
    {"t", sizeof(double)},
    {"x", sizeof(double)},
    {"y", sizeof(double)},
    {"z", sizeof(double)},
    {"mass", sizeof(double)},
    {"p0", sizeof(double)},
    {"px", sizeof(double)},
    {"py", sizeof(double)},
    {"pz", sizeof(double)},

    {"pdg", sizeof(int32_t)},
    {"id", sizeof(int32_t)},
    {"charge", sizeof(int32_t)},
    {"ncoll", sizeof(int32_t)},

    {"form_time", sizeof(double)},
    {"xsecfac", sizeof(double)},
    {"proc_id_origin", sizeof(int32_t)},
    {"proc_type_origin", sizeof(int32_t)},
    {"time_last_coll", sizeof(double)},
    {"pdg_mother1", sizeof(int32_t)},
    {"pdg_mother2", sizeof(int32_t)},
    {"baryon_number", sizeof(int32_t)},
    {"strangeness", sizeof(int32_t)},

    {"tau", sizeof(double)},
    {"eta", sizeof(double)},
    {"eta_s", sizeof(double)},
    {"mt", sizeof(double)},
    {"Rap", sizeof(double)},
    {"y_rap", sizeof(double)},
    {"perturbative_weight", sizeof(double)},
};

inline std::vector<QuantityOffset> quantity_offsets_from_quantities(
    const std::vector<std::string>& quantities) {
    std::vector<QuantityOffset> offsets;
    offsets.reserve(quantities.size());

    std::size_t offset = 0;

    for (const auto& q : quantities) {
        const auto it = quantity_sizes.find(q);

        if (it == quantity_sizes.end()) {
            throw std::runtime_error("unknown quantity: " + q);
        }

        const std::size_t size = it->second;

        offsets.push_back(QuantityOffset{
            .name = q,
            .offset = offset,
            .size = size,
        });

        offset += size;
    }

    return offsets;
}

inline std::size_t particle_size_from_quantities(
    const std::vector<std::string>& quantities) {
    const auto offsets = quantity_offsets_from_quantities(quantities);

    if (offsets.empty()) {
        return 0;
    }

    const auto& last = offsets.back();
    return last.offset + last.size;
}

inline ParticleLayout particle_layout_from_quantities(
    const std::vector<std::string>& quantities) {
    ParticleLayout layout;
    layout.offsets = quantity_offsets_from_quantities(quantities);
    layout.particle_size = particle_size_from_quantities(quantities);
    return layout;
}

}  // namespace brass

#endif  // BRASS_QUANTITIES_H
