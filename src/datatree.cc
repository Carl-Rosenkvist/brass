#include "datatree.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ---- YAML for Data ----
void to_yaml(YAML::Emitter &out, const Data &v) {
    std::visit(
        [&](const auto &x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
            } else if constexpr (std::is_same_v<T, int> ||
                                 std::is_same_v<T, double>) {
                out << x;
            } else if constexpr (std::is_same_v<T, std::vector<int>> ||
                                 std::is_same_v<T, std::vector<double>>) {
                out << YAML::BeginSeq;
                for (const auto &e : x) out << e;
                out << YAML::EndSeq;
            } else if constexpr (std::is_same_v<T, Histogram1D>) {
                to_yaml(out, x);
            } else {
                static_assert(sizeof(T) == 0,
                              "Unhandled Data type in to_yaml(Data)");
            }
        },
        v);
}

// ---- YAML for DataNode ----

void to_yaml(YAML::Emitter &out, const DataNode &node) {
    if (node.get_name().empty()) {
        // Root node: emit children only
        out << YAML::BeginMap;
        for (const auto &[_, child] : node.children()) {
            to_yaml(out, child);
        }
        out << YAML::EndMap;
        return;
    }

    if (node.is_leaf()) {
        out << YAML::Key << node.get_name();
        to_yaml(out, node.get_data());
    } else {
        out << YAML::Key << node.get_name();
        out << YAML::Value << YAML::BeginMap;
        for (auto &[_, child] : node.children()) {
            to_yaml(out, child);
        }
        out << YAML::EndMap;
    }
}
namespace {
// tolerances for doubles
struct Tol {
    double rtol = 1e-6;
    double atol = 1e-9;
};

inline bool approx_equal(double a, double b, const Tol &t) {
    const double diff = std::fabs(a - b);
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= (t.atol + t.rtol * scale);
}

// visit helper
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

inline std::string path_join(const std::string &p, const std::string &k) {
    if (p.empty()) return k;
    if (k.empty()) return p;
    return p + "/" + k;
}
}  // namespace

// ---------- free function: merge_values (Data <- Data) ----------
void merge_values(Data &a, const Data &b, const std::string &path) {
    // empty rules
    if (std::holds_alternative<std::monostate>(a)) {
        a = b;
        return;
    }
    if (std::holds_alternative<std::monostate>(b)) {
        return;
    }

    const Tol tol{};
    std::visit(
        overloaded{
            // scalars
            [&](int &av, int bv) { av += bv; },
            [&](double &av, double bv) { av += bv; },

            // vectors: append
            [&](std::vector<int> &av, const std::vector<int> &bv) {
                av.insert(av.end(), bv.begin(), bv.end());
            },
            [&](std::vector<double> &av, const std::vector<double> &bv) {
                av.insert(av.end(), bv.begin(), bv.end());
            },

            // histogram: +=
            [&](Histogram1D &av, const Histogram1D &bv) { av += bv; },

            // disallowed mixes (no int<->double or vec<int><->vec<double>)
            [&](int &, double) {
                throw std::runtime_error("type mix int/double at '" + path +
                                         "'");
            },
            [&](double &, int) {
                throw std::runtime_error("type mix double/int at '" + path +
                                         "'");
            },
            [&](std::vector<int> &, const std::vector<double> &) {
                throw std::runtime_error("type mix vec<int>/vec<double> at '" +
                                         path + "'");
            },
            [&](std::vector<double> &, const std::vector<int> &) {
                throw std::runtime_error("type mix vec<double>/vec<int> at '" +
                                         path + "'");
            },

            // any other mismatch
            [&](auto &, const auto &) {
                throw std::runtime_error("type mismatch at '" + path + "'");
            }},
        a, b);
}

DataNode &DataNode::operator+=(const DataNode &other) {
    // Recursive node merge
    std::function<void(DataNode &, const DataNode &, const std::string &)>
        merge_nodes;

    merge_nodes = [&](DataNode &dst, const DataNode &src,
                      const std::string &path) {
        const bool dv = dst.has_value();
        const bool sv = src.has_value();
        const bool dc = !dst.subdata.empty();
        const bool sc = !src.subdata.empty();

        // schema sanity
        if (dv && dc)
            throw std::runtime_error("invalid schema at '" + path +
                                     "': both value and children");
        if (sv && sc)
            throw std::runtime_error("invalid source schema at '" + path +
                                     "': both value and children");

        // leaf vs internal is a hard conflict
        if ((dv && sc) || (dc && sv))
            throw std::runtime_error("schema conflict at '" + path + "'");

        // leaf + leaf → delegate to free merge_values
        if (dv && sv) {
            ::merge_values(dst.value, src.value, path);
            return;
        }

        // internal + internal → merge children by key
        for (const auto &[k, schild] : src.subdata) {
            auto it = dst.subdata.find(k);
            if (it == dst.subdata.end()) {
                dst.subdata.emplace(k, schild);  // copy subtree
            } else {
                merge_nodes(it->second, schild, path_join(path, k));
            }
        }
    };

    merge_nodes(*this, other, name.empty() ? std::string{"<root>"} : name);
    return *this;
}
