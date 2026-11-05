#ifndef BRASS_BLOCK_H
#define BRASS_BLOCK_H

#include <array>
#include <cstdint>
#include <stdexcept>
#include <optional>
#include <string>
#include <string_view>
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

enum class HeaderQuantity : std::uint8_t { format_version, format_variant };

enum class EndBlockQuantity : std::uint8_t {
    event_number,
    ensemble_number,
    impact_parameter,
    empty,
};

enum class InteractionQuantity : std::uint8_t {
    n_in,
    n_out,
    rho,
    sigma,
    sigma_p,
    process,
};

inline constexpr std::array<NamedValue<HeaderQuantity>, 2> header_quantity_names{{
    {"format_version", HeaderQuantity::format_version},
    {"format_variant", HeaderQuantity::format_variant},
}};

inline constexpr std::array<NamedValue<EndBlockQuantity>, 4>
    end_block_quantity_names{{
        {"event_number", EndBlockQuantity::event_number},
        {"ensemble_number", EndBlockQuantity::ensemble_number},
        {"impact_parameter", EndBlockQuantity::impact_parameter},
        {"empty", EndBlockQuantity::empty},
    }};

inline constexpr std::array<NamedValue<InteractionQuantity>, 10>
    interaction_quantity_names{{
        {"n_in", InteractionQuantity::n_in},
        {"n_out", InteractionQuantity::n_out},
        {"rho", InteractionQuantity::rho},
        {"density", InteractionQuantity::rho},
        {"sigma", InteractionQuantity::sigma},
        {"weight", InteractionQuantity::sigma},
        {"sigma_p", InteractionQuantity::sigma_p},
        {"partial_weight", InteractionQuantity::sigma_p},
        {"process", InteractionQuantity::process},
        {"process_type", InteractionQuantity::process},
    }};

inline HeaderQuantity parse_header_quantity(std::string_view name) {
    if (const auto q = find_by_name(header_quantity_names, name)) {
        return *q;
    }

    throw std::runtime_error("unknown header quantity: " + std::string(name));
}

inline EndBlockQuantity parse_end_block_quantity(std::string_view name) {
    if (const auto q = find_by_name(end_block_quantity_names, name)) {
        return *q;
    }

    throw std::runtime_error("unknown end block quantity: " +
                             std::string(name));
}

inline std::optional<InteractionQuantity> try_parse_interaction_quantity(
    std::string_view name) {
    return find_by_name(interaction_quantity_names, name);
}

inline InteractionQuantity parse_interaction_quantity(std::string_view name) {
    if (const auto q = try_parse_interaction_quantity(name)) {
        return *q;
    }

    throw std::runtime_error("unknown interaction quantity: " +
                             std::string(name));
}

inline double header_value(const Header& header, HeaderQuantity q) {
    switch (q) {
        case HeaderQuantity::format_version:
            return static_cast<double>(header.format_version);
        case HeaderQuantity::format_variant:
            return static_cast<double>(header.format_variant);
    }

    throw std::logic_error("unhandled header quantity");
}

inline double end_block_value(const EndBlock& block, EndBlockQuantity q) {
    switch (q) {
        case EndBlockQuantity::event_number:
            return static_cast<double>(block.event_number);
        case EndBlockQuantity::ensemble_number:
            return static_cast<double>(block.ensemble_number);
        case EndBlockQuantity::impact_parameter:
            return block.impact_parameter;
        case EndBlockQuantity::empty:
            return block.empty ? 1.0 : 0.0;
    }

    throw std::logic_error("unhandled end block quantity");
}

inline double interaction_value(const InteractionBlock& block,
                                InteractionQuantity q) {
    switch (q) {
        case InteractionQuantity::n_in:
            return static_cast<double>(block.n_in);
        case InteractionQuantity::n_out:
            return static_cast<double>(block.n_out);
        case InteractionQuantity::rho:
            return block.rho;
        case InteractionQuantity::sigma:
            return block.sigma;
        case InteractionQuantity::sigma_p:
            return block.sigma_p;
        case InteractionQuantity::process:
            return static_cast<double>(block.process);
    }

    throw std::logic_error("unhandled interaction quantity");
}
}  // namespace brass

#endif  // BRASS_BLOCK_H
