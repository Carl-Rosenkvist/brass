#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

enum class QuantityType { Double, Int32 };

size_t type_size(QuantityType t);

// name -> type
extern const std::unordered_map<std::string, QuantityType> quantity_string_map;

// name -> byte offset within a particle record (computed from names order)
std::unordered_map<std::string, size_t>
compute_quantity_layout(const std::vector<std::string>& names);

// Read exactly `size` bytes from stream or throw
std::vector<char> read_chunk(std::ifstream& bfile, size_t size);

template <typename T>
inline T get_quantity(std::span<const char> particle,
                      const std::string& name,
                      const std::unordered_map<std::string, size_t>& layout) {
    auto it_info = quantity_string_map.find(name);
    if (it_info == quantity_string_map.end())
        throw std::runtime_error("Unknown quantity: " + name);

    if constexpr (std::is_same_v<T, double>) {
        if (it_info->second != QuantityType::Double)
            throw std::runtime_error("Requested double, but quantity is not double: " + name);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        if (it_info->second != QuantityType::Int32)
            throw std::runtime_error("Requested int32, but quantity is not int32: " + name);
    } else {
        static_assert(!sizeof(T*), "Unsupported T in get_quantity");
    }

    auto it = layout.find(name);
    if (it == layout.end())
        throw std::runtime_error("Quantity not in layout: " + name);

    const size_t offset = it->second;
    if (offset + sizeof(T) > particle.size())
        throw std::runtime_error("Buffer too small for " + name);

    T value;
    std::memcpy(&value, particle.data() + offset, sizeof(T));
    return value;
}
