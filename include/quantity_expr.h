#ifndef BRASS_QUANTITY_EXPR_H
#define BRASS_QUANTITY_EXPR_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "particles.h"

namespace brass {

enum class QuantityKind {
    Column,
    Pt,
    P,
    Mass,
    Mt,
    Rapidity,
};

struct QuantityExpr {
    QuantityKind kind = QuantityKind::Column;
    std::string name;
};

inline QuantityExpr quantity_expr_from_name(const std::string& name) {
    if (name == "pt") {
        return {QuantityKind::Pt, name};
    }

    if (name == "p") {
        return {QuantityKind::P, name};
    }

    if (name == "m" || name == "mass_from_momentum") {
        return {QuantityKind::Mass, name};
    }

    if (name == "mt") {
        return {QuantityKind::Mt, name};
    }

    if (name == "y" || name == "y_rap" || name == "rapidity") {
        return {QuantityKind::Rapidity, name};
    }

    return {QuantityKind::Column, name};
}

inline double evaluate_quantity(const QuantityExpr& expr,
                                const Particles& particles, std::size_t i) {
    switch (expr.kind) {
        case QuantityKind::Column: {
            return particles.column<double>(expr.name)[i];
        }

        case QuantityKind::Pt: {
            const auto& px = particles.column<double>("px");
            const auto& py = particles.column<double>("py");

            return std::hypot(px[i], py[i]);
        }

        case QuantityKind::P: {
            const auto& px = particles.column<double>("px");
            const auto& py = particles.column<double>("py");
            const auto& pz = particles.column<double>("pz");

            return std::hypot(std::hypot(px[i], py[i]), pz[i]);
        }

        case QuantityKind::Mass: {
            const auto& p0 = particles.column<double>("p0");
            const auto& px = particles.column<double>("px");
            const auto& py = particles.column<double>("py");
            const auto& pz = particles.column<double>("pz");

            const double m2 =
                p0[i] * p0[i] - px[i] * px[i] - py[i] * py[i] - pz[i] * pz[i];

            return std::sqrt(std::max(m2, 0.0));
        }

        case QuantityKind::Mt: {
            const auto& p0 = particles.column<double>("p0");
            const auto& px = particles.column<double>("px");
            const auto& py = particles.column<double>("py");
            const auto& pz = particles.column<double>("pz");

            const double pt = std::hypot(px[i], py[i]);

            const double m2 =
                p0[i] * p0[i] - px[i] * px[i] - py[i] * py[i] - pz[i] * pz[i];

            const double mass = std::sqrt(std::max(m2, 0.0));

            return std::hypot(pt, mass);
        }

        case QuantityKind::Rapidity: {
            const auto& p0 = particles.column<double>("p0");
            const auto& pz = particles.column<double>("pz");

            if (p0[i] <= std::abs(pz[i])) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            return 0.5 * std::log((p0[i] + pz[i]) / (p0[i] - pz[i]));
        }
    }

    throw std::runtime_error("unknown quantity expression");
}

}  // namespace brass

#endif  // BRASS_QUANTITY_EXPR_H
