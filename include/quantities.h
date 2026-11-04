#ifndef BRASS_QUANTITIES_H
#define BRASS_QUANTITIES_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brass {

// Every per-particle quantity BRASS knows about. Names (strings) are only
// used at the boundary: they are parsed into this enum once, and all lookups
// afterwards are by enum.
enum class Quantity : std::uint8_t {
    // Quantities SMASH can write to binary output
    t,
    x,
    y,
    z,
    mass,
    p0,
    px,
    py,
    pz,
    pdg,
    id,
    charge,
    ncoll,
    form_time,
    xsecfac,
    proc_id_origin,
    proc_type_origin,
    time_last_coll,
    pdg_mother1,
    pdg_mother2,
    baryon_number,
    strangeness,
    tau,
    eta,
    eta_s,
    mt,
    y_rap,
    perturbative_weight,

    // Quantities that can only be computed
    pt,
    p,

    Count,
};

inline constexpr std::size_t quantity_count =
    static_cast<std::size_t>(Quantity::Count);

enum class QuantityType : std::uint8_t { Double, Int32 };

inline constexpr std::size_t quantity_type_size(QuantityType type) {
    return type == QuantityType::Double ? sizeof(double) : sizeof(int32_t);
}

// Name table for a quantity enum. Several names may map to the same value
// (aliases); the first entry for a value is its canonical name.
template <class E>
struct NamedValue {
    std::string_view name;
    E value;
};

template <class E, std::size_t N>
constexpr std::optional<E> find_by_name(
    const std::array<NamedValue<E>, N>& table, std::string_view name) {
    for (const auto& entry : table) {
        if (entry.name == name) {
            return entry.value;
        }
    }

    return std::nullopt;
}

template <class E, std::size_t N>
constexpr std::string_view find_name(const std::array<NamedValue<E>, N>& table,
                                     E value) {
    for (const auto& entry : table) {
        if (entry.value == value) {
            return entry.name;
        }
    }

    return "<unknown>";
}

inline constexpr std::array<NamedValue<Quantity>, 32> quantity_names{{
    {"t", Quantity::t},
    {"x", Quantity::x},
    {"y", Quantity::y},
    {"z", Quantity::z},
    {"mass", Quantity::mass},
    {"p0", Quantity::p0},
    {"px", Quantity::px},
    {"py", Quantity::py},
    {"pz", Quantity::pz},
    {"pdg", Quantity::pdg},
    {"id", Quantity::id},
    {"charge", Quantity::charge},
    {"ncoll", Quantity::ncoll},
    {"form_time", Quantity::form_time},
    {"xsecfac", Quantity::xsecfac},
    {"proc_id_origin", Quantity::proc_id_origin},
    {"proc_type_origin", Quantity::proc_type_origin},
    {"time_last_coll", Quantity::time_last_coll},
    {"pdg_mother1", Quantity::pdg_mother1},
    {"pdg_mother2", Quantity::pdg_mother2},
    {"baryon_number", Quantity::baryon_number},
    {"strangeness", Quantity::strangeness},
    {"tau", Quantity::tau},
    {"eta", Quantity::eta},
    {"eta_s", Quantity::eta_s},
    {"mt", Quantity::mt},
    {"y_rap", Quantity::y_rap},
    {"Rap", Quantity::y_rap},
    {"rapidity", Quantity::y_rap},
    {"perturbative_weight", Quantity::perturbative_weight},
    {"pt", Quantity::pt},
    {"p", Quantity::p},
}};

inline std::optional<Quantity> try_parse_quantity(std::string_view name) {
    return find_by_name(quantity_names, name);
}

inline Quantity parse_quantity(std::string_view name) {
    if (const auto q = try_parse_quantity(name)) {
        return *q;
    }

    throw std::runtime_error("unknown particle quantity: " + std::string(name));
}

inline std::string_view to_string(Quantity q) {
    return find_name(quantity_names, q);
}

// Storage type of a quantity in the binary file. Only meaningful for
// quantities SMASH can write (is_stored_quantity).
inline constexpr QuantityType quantity_type(Quantity q) {
    switch (q) {
        case Quantity::pdg:
        case Quantity::id:
        case Quantity::charge:
        case Quantity::ncoll:
        case Quantity::proc_id_origin:
        case Quantity::proc_type_origin:
        case Quantity::pdg_mother1:
        case Quantity::pdg_mother2:
        case Quantity::baryon_number:
        case Quantity::strangeness:
            return QuantityType::Int32;

        default:
            return QuantityType::Double;
    }
}

inline constexpr bool is_stored_quantity(Quantity q) {
    return q != Quantity::pt && q != Quantity::p && q != Quantity::Count;
}

struct QuantityOffset {
    Quantity quantity = Quantity::Count;
    QuantityType type = QuantityType::Double;
    std::string name;  // name as written in the quantity list (for display)
    std::size_t offset = 0;
    std::size_t size = 0;
};

struct ParticleLayout {
    std::vector<QuantityOffset> offsets;  // in file order
    std::array<std::optional<std::size_t>, quantity_count> index{};
    std::size_t particle_size = 0;

    const QuantityOffset* find(Quantity q) const {
        const auto& i = index[static_cast<std::size_t>(q)];
        return i ? &offsets[*i] : nullptr;
    }

    const QuantityOffset& at(Quantity q) const {
        if (const auto* offset = find(q)) {
            return *offset;
        }

        throw std::runtime_error("particle quantity not in file: " +
                                 std::string(to_string(q)));
    }
};

inline ParticleLayout particle_layout_from_quantities(
    const std::vector<std::string>& quantities) {
    ParticleLayout layout;
    layout.offsets.reserve(quantities.size());

    std::size_t offset = 0;

    for (const auto& name : quantities) {
        const auto q = try_parse_quantity(name);

        if (!q || !is_stored_quantity(*q)) {
            throw std::runtime_error("unknown quantity: " + name);
        }

        auto& slot = layout.index[static_cast<std::size_t>(*q)];

        if (slot) {
            throw std::runtime_error("duplicate quantity: " + name);
        }

        const auto type = quantity_type(*q);
        const auto size = quantity_type_size(type);

        slot = layout.offsets.size();
        layout.offsets.push_back(QuantityOffset{
            .quantity = *q,
            .type = type,
            .name = name,
            .offset = offset,
            .size = size,
        });

        offset += size;
    }

    layout.particle_size = offset;
    return layout;
}

inline std::size_t particle_size_from_quantities(
    const std::vector<std::string>& quantities) {
    return particle_layout_from_quantities(quantities).particle_size;
}

}  // namespace brass

#endif  // BRASS_QUANTITIES_H
