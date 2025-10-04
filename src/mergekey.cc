#include "mergekey.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>


// ---- MergeKey implementation ----
MergeKey::MergeKey(std::string n, MergeKeyValue v)
    : name(std::move(n)),
      value(std::visit([](auto x) -> MergeKeyValue {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T,double>) {
              double s = 1000.0; // round to 3 decimals
              return std::round(x * s) / s;
          } else {
              return x;
          }
      }, v)) {}

bool operator<(MergeKey const& a, MergeKey const& b) {
    if (a.name != b.name) return a.name < b.name;
    return a.value < b.value; // variant orders by index, then value
}

bool operator==(MergeKey const& a, MergeKey const& b) {
    return a.name == b.name && a.value == b.value;
}

bool operator<(MergeKeySet const& A, MergeKeySet const& B) {
    return std::lexicographical_compare(A.begin(), A.end(), B.begin(), B.end());
}

bool operator==(MergeKeySet const& A, MergeKeySet const& B) {
    return A.size() == B.size() && std::equal(A.begin(), A.end(), B.begin());
}

// ---- YAML emitters ----
void to_yaml(YAML::Emitter& out, const MergeKeyValue& v) {
    std::visit([&](auto const& x){
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::string>) {
            out << YAML::DoubleQuoted << x;
        } else {
            out << x;
        }
    }, v);
}

void to_yaml(YAML::Emitter& out, MergeKey const& mk) {
    out << YAML::BeginMap;
    out << YAML::Key << mk.name << YAML::Value;
    to_yaml(out, mk.value);
    out << YAML::EndMap;
}

void to_yaml(YAML::Emitter& out, MergeKeySet const& set) {
    out << YAML::BeginMap;
    for (auto const& mk : set) {
        out << YAML::Key << mk.name << YAML::Value;
        to_yaml(out, mk.value);
    }
    out << YAML::EndMap;
}


MergeKeySet parse_merge_key(const std::string& meta) {
    MergeKeySet ks;
    if (meta.empty()) return ks;

    std::stringstream ss(meta);
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto eq = item.find('=');
        if (eq == std::string::npos) continue;
        std::string name = item.substr(0, eq);
        std::string val  = item.substr(eq + 1);

        try {
            if (val.find('.') != std::string::npos) {
                ks.emplace_back(name, std::stod(val));
            } else {
                try {
                    ks.emplace_back(name, std::stoi(val));
                } catch (...) {
                    ks.emplace_back(name, std::stod(val));
                }
            }
        } catch (...) {
            ks.emplace_back(name, val);
        }
    }
    sort_keyset(ks);
    return ks;
}

void sort_keyset(MergeKeySet& k) {
    std::sort(k.begin(), k.end(), [](auto const& a, auto const& b){
        if (a.name != b.name) return a.name < b.name;
        return a.value < b.value;
    });
}

bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}


std::string label_from_keyset(const MergeKeySet& ks) {
    std::ostringstream oss;
    bool first = true;
    for (auto const& mk : ks) {
        if (!first) {
            oss << "_";
        }
        first = false;
        oss << mk.name << "-";
        std::visit([&oss](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, double>) {
                // fixed-point with trimming
                oss << std::setprecision(6) << std::fixed << arg;
            } else {
                oss << arg;
            }
        }, mk.value);
    }
    return oss.str();
}

