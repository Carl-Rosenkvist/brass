#ifndef BRASS_BLOCK_H
#define BRASS_BLOCK_H

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

#include "particles.h"

namespace brass {

struct Header {
    std::array<char, 5> magic_number{{0, 0, 0, 0, 0}};  // includes NUL
    uint16_t format_version = 0;
    uint16_t format_variant = 0;
    std::string smash_version;
};

struct ParticleBlock {
    int32_t event_number = 0;
    int32_t ensemble_number = 0;
    Particles particles;
};

struct EndBlock {
    uint32_t event_number = 0;
    int32_t ensemble_number = 0;
    double impact_parameter = 0.0;
    bool empty = false;
};

struct InteractionBlock {
    int32_t n_in = 0;
    int32_t n_out = 0;
    double rho = 0.0;
    double sigma = 0.0;
    double sigma_p = 0.0;
    int32_t process = 0;

    Particles incoming;
    Particles outgoing;
};

using Block = std::variant<ParticleBlock, EndBlock, InteractionBlock>;

inline double header_value(const Header& header, const std::string& name) {
    if (name == "format_version") {
        return static_cast<double>(header.format_version);
    }

    if (name == "format_variant") {
        return static_cast<double>(header.format_variant);
    }

    throw std::runtime_error("unknown header quantity: " + name);
}

inline double end_block_value(const EndBlock& block, const std::string& name) {
    if (name == "event_number") {
        return static_cast<double>(block.event_number);
    }

    if (name == "ensemble_number") {
        return static_cast<double>(block.ensemble_number);
    }

    if (name == "impact_parameter") {
        return block.impact_parameter;
    }

    if (name == "empty") {
        return block.empty ? 1.0 : 0.0;
    }

    throw std::runtime_error("unknown end block quantity: " + name);
}

inline bool is_interaction_quantity(const std::string& name) {
    return name == "n_in" || name == "n_out" || name == "rho" ||
           name == "density" || name == "sigma" || name == "weight" ||
           name == "sigma_p" || name == "partial_weight" || name == "process" ||
           name == "process_type";
}

inline double interaction_value(const InteractionBlock& block,
                                const std::string& name) {
    if (name == "n_in") {
        return static_cast<double>(block.n_in);
    }

    if (name == "n_out") {
        return static_cast<double>(block.n_out);
    }

    if (name == "rho" || name == "density") {
        return block.rho;
    }

    if (name == "sigma" || name == "weight") {
        return block.sigma;
    }

    if (name == "sigma_p" || name == "partial_weight") {
        return block.sigma_p;
    }

    if (name == "process" || name == "process_type") {
        return static_cast<double>(block.process);
    }

    throw std::runtime_error("unknown interaction quantity: " + name);
}
}  // namespace brass

#endif  // BRASS_BLOCK_H
