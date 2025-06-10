#pragma once
#include <variant>
#include <vector>
#include <map>
#include <string>
#include <type_traits>
#include "histogram1d.h"

#include <yaml-cpp/yaml.h>
using Data = std::variant<std::monostate, int, double,
                          std::vector<int>, std::vector<double>, Histogram1D>;
void merge_values(Data& a, const Data& b, const std::string& path);
 // YAML serialization



// datanode.hpp (or wherever DataNode lives)

class DataNode {
public:
  DataNode() : name(""), value(std::monostate{}) {}
  explicit DataNode(const std::string& name) : name(name) {}

  template<typename T>
  DataNode(const std::string& name, T&& value)
  : name(name) {
    static_assert(std::is_constructible_v<Data, T&&>,
                  "Type not supported by Data variant");
    this->value = Data(std::forward<T>(value));
  }

  // Accessors (public)
  const std::string& get_name() const { return name; }
  const Data& get_data() const { return value; }
  Data&       get_data()       { return value; }
  DataNode& operator+=(const DataNode& other); 

  std::map<std::string, DataNode>&       children()       { return subdata; }
  const std::map<std::string, DataNode>& children() const { return subdata; }

 
DataNode& add_child(const std::string& key) {
    value = std::monostate{}; // clear any accidental scalar
    auto [it, inserted] = subdata.emplace(key, DataNode(key));
    return it->second;
}

 
template <typename T>
DataNode& add_child(const std::string& key, T&& value_in) {
    auto [it, inserted] = subdata.emplace(key, DataNode(key, std::forward<T>(value_in)));
    return it->second;
}
  bool has_value() const { return !std::holds_alternative<std::monostate>(value); }
  bool is_leaf()   const { return has_value() && subdata.empty(); }
  bool empty()     const { return !has_value() && subdata.empty(); } // convenience

private:
  std::string name = "";
  Data value{};
  std::map<std::string, DataNode> subdata;
};

void to_yaml(YAML::Emitter& out, const Data& v);
void to_yaml(YAML::Emitter& out, const DataNode& v);


