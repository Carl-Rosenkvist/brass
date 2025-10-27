#include "quantities.h"

const std::unordered_map<std::string, QuantityType> quantity_string_map = {
    {"t", QuantityType::Double},
    {"x", QuantityType::Double},
    {"y", QuantityType::Double},
    {"z", QuantityType::Double},
    {"mass", QuantityType::Double},
    {"p0", QuantityType::Double},
    {"px", QuantityType::Double},
    {"py", QuantityType::Double},
    {"pz", QuantityType::Double},
    {"pdg", QuantityType::Int32},
    {"id", QuantityType::Int32},
    {"charge", QuantityType::Int32},
    {"ncoll", QuantityType::Int32},
    {"form_time", QuantityType::Double},
    {"xsecfac", QuantityType::Double},
    {"proc_id_origin", QuantityType::Int32},
    {"proc_type_origin", QuantityType::Int32},
    {"time_last_coll", QuantityType::Double},
    {"pdg_mother1", QuantityType::Int32},
    {"pdg_mother2", QuantityType::Int32},
    {"baryon_number", QuantityType::Int32},
    {"strangeness", QuantityType::Int32},
};

// --- Helpers ---

size_t type_size(QuantityType t) {
    switch (t) {
        case QuantityType::Double:
            return sizeof(double);
        case QuantityType::Int32:
            return sizeof(int32_t);
    }
    throw std::logic_error("Unknown QuantityType");
}

std::unordered_map<std::string, size_t> compute_quantity_layout(
    const std::vector<std::string> &names) {
    std::unordered_map<std::string, size_t> layout;
    size_t offset = 0;

    for (const auto &name : names) {
        auto it = quantity_string_map.find(name);
        if (it == quantity_string_map.end())
            throw std::runtime_error("Unknown quantity: " + name);

        layout[name] = offset;
        offset += type_size(it->second);
    }

    return layout;
}
