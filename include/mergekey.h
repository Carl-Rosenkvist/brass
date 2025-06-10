#ifndef MERGEKEY_H
#define MERGEKEY_H

#include <yaml-cpp/yaml.h>
#include <variant>
#include <string>
#include <vector>

// A merge key can hold int, double, or string
using MergeKeyValue = std::variant<int, double, std::string>;

struct MergeKey {
    std::string name;
    MergeKeyValue value;

    explicit MergeKey(std::string n, MergeKeyValue v);
};

// Comparisons for MergeKey
bool operator<(MergeKey const& a, MergeKey const& b);
bool operator==(MergeKey const& a, MergeKey const& b);

using MergeKeySet = std::vector<MergeKey>;

// Comparisons for MergeKeySet
bool operator<(MergeKeySet const& A, MergeKeySet const& B);
bool operator==(MergeKeySet const& A, MergeKeySet const& B);

// YAML emitters
void to_yaml(YAML::Emitter& out, const MergeKeyValue& v);
void to_yaml(YAML::Emitter& out, MergeKey const& mk);
void to_yaml(YAML::Emitter& out, MergeKeySet const& set);


MergeKeySet parse_merge_key(const std::string& meta);
void sort_keyset(MergeKeySet& k);
bool ends_with(const std::string& str, const std::string& suffix);
std::string label_from_keyset(const MergeKeySet& ks);


#endif // MERGEKEY_H
