// BinaryReader.h
#ifndef BINARY_READER_H
#define BINARY_READER_H

#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <span>
#include <optional>
// Enum classes and helper structures

enum class Quantity {
    T, X, Y, Z,               // space-time coordinates
    MASS,                     // rest-mass
    P0, PX, PY, PZ,           // energy and 3-momentum
    PDG,                      // PDG code
    ID,                        // identifier
    CHARGE,                   // electric charge
    NCOLL,                    // number of collisions
    FORM_TIME,                // formation time
    XSEC_FACTOR,              // cross section scaling factor
    PROC_ID_ORIGIN,           // ID of last process
    PROC_TYPE_ORIGIN,         // type of last process
    TIME_LAST_COLL,           // time of last interaction
    PDG_MOTHER1,              // PDG code of mother 1
    PDG_MOTHER2,              // PDG code of mother 2
    BARYON_NUMBER,            // baryon number
    STRANGENESS,              // strangeness
};


enum class QuantityType {
    Double,
    Int32
};

struct QuantityInfo {
    Quantity quantity;
    QuantityType type;
};

size_t type_size(QuantityType t);

extern const std::unordered_map<std::string, QuantityInfo> quantity_string_map;

std::unordered_map<Quantity, size_t>
compute_quantity_layout(const std::vector<std::string>& names);

std::vector<char> read_chunk(std::ifstream& bfile, size_t size);

// Template helpers

template <typename T>
T extract_and_advance(const std::vector<char>& buffer, size_t& offset) {
    T value;
    std::memcpy(&value, buffer.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}


template<typename T>
T get_quantity(std::span<const char> particle,
               const std::string& name,
               const std::unordered_map<Quantity, size_t>& layout)
{
    auto it_info = quantity_string_map.find(name);
    if (it_info == quantity_string_map.end())
        throw std::runtime_error("Unknown quantity: " + name);

    if (std::is_same_v<T, double> && it_info->second.type != QuantityType::Double)
        throw std::runtime_error("Requested double, but quantity is not double: " + name);
    if (std::is_same_v<T, int32_t> && it_info->second.type != QuantityType::Int32)
        throw std::runtime_error("Requested int32, but quantity is not int32: " + name);

    Quantity q = it_info->second.quantity;
    auto it = layout.find(q);
    if (it == layout.end())
        throw std::runtime_error("Quantity not in layout: " + name);

    size_t offset = it->second;
    if (offset + sizeof(T) > particle.size())
        throw std::runtime_error("Buffer too small for " + name);

    T value;
    std::memcpy(&value, particle.data() + offset, sizeof(T));
    return value;
}

class Header {
public:
    const std::array<char, 5> magic_number; // includes NUL
    const uint16_t            format_version;
    const uint16_t            format_variant;
    const std::string         smash_version;

    // Factory that fully initializes all const members
    static Header read_from(std::ifstream& bfile) {
        std::array<char, 5> magic_buf{{0,0,0,0,0}};
        bfile.read(magic_buf.data(), 4);
        magic_buf[4] = '\0';

        uint16_t version = 0;
        uint16_t variant = 0;
        bfile.read(reinterpret_cast<char*>(&version), sizeof(version));
        bfile.read(reinterpret_cast<char*>(&variant), sizeof(variant));

        uint32_t len = 0;
        bfile.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!bfile) {
            throw std::runtime_error("Failed to read header (length).");
        }

        std::string smash_ver;
        if (len > 0) {
            std::vector<char> buf(len);
            bfile.read(buf.data(), len);
            smash_ver.assign(buf.begin(), buf.end());
        }

        if (!bfile) {
            throw std::runtime_error("Failed to read header contents.");
        }

        return Header(magic_buf, version, variant, std::move(smash_ver));
    }

    void print() const {
        std::cout << "Magic Number:   " << magic_number.data() << "\n"
                  << "Format Version: " << format_version      << "\n"
                  << "Format Variant: " << format_variant      << "\n"
                  << "Smash Version:  " << smash_version       << "\n";
    }

private:
    // Private ctor ensures all const members are initialized in the init list
    Header(std::array<char,5> magic,
           uint16_t version,
           uint16_t variant,
           std::string smash_ver)
        : magic_number(magic),
          format_version(version),
          format_variant(variant),
          smash_version(std::move(smash_ver)) {}
};
class EndBlock {
  public:
    const uint32_t event_number;
    const uint32_t ensamble_number;   
    const double   impact_parameter;
    const bool     empty;

    // On-disk layout: u32, u32, double, u8
    static constexpr size_t SIZE = 4u + 4u + 8u + 1u;

    // Read from stream and return a fully-constructed EndBlock
    static EndBlock read_from(std::ifstream& bfile) {
        std::vector<char> buffer = read_chunk(bfile, SIZE);
        size_t offset = 0;

        uint32_t ev  = extract_and_advance<uint32_t>(buffer, offset);
        uint32_t ens = extract_and_advance<uint32_t>(buffer, offset);
        double   b   = extract_and_advance<double>(buffer, offset);
        uint8_t  raw = extract_and_advance<uint8_t>(buffer, offset);

        bool emp = (raw != 0); // nonzero byte => true

        return EndBlock(ev, ens, b, emp);
    }

private:
    EndBlock(uint32_t ev, uint32_t ens, double b, bool emp)
        : event_number(ev), ensamble_number(ens), impact_parameter(b), empty(emp) {}
};

struct ParticleBlock {
    const int32_t  event_number;
    const int32_t  ensamble_number;  // keep spelling if file format uses it
    const uint32_t npart;
    const size_t   particle_size;
    const std::vector<char> particles;

    // Value constructor: only initializes (no reading here)
    ParticleBlock(int32_t ev, int32_t ens, uint32_t n, size_t psize, std::vector<char> data)
      : event_number(ev),
        ensamble_number(ens),
        npart(n),
        particle_size(psize),
        particles(std::move(data)) {}

    // Factory: reads from stream, then constructs via the value constructor
    static ParticleBlock read_from(std::ifstream& bfile, size_t psize) {
        constexpr size_t HEADER_SIZE = sizeof(int32_t) + sizeof(int32_t) + sizeof(uint32_t);
        auto header = read_chunk(bfile, HEADER_SIZE);

        size_t off = 0;
        const int32_t  ev  = extract_and_advance<int32_t>(header, off);
        const int32_t  ens = extract_and_advance<int32_t>(header, off);
        const uint32_t n   = extract_and_advance<uint32_t>(header, off);

        // size checks
        const size_t bytes = static_cast<size_t>(n) * psize;
        if (psize != 0 && bytes / psize != n)
            throw std::runtime_error("size overflow in ParticleBlock");

        auto data = read_chunk(bfile, bytes);
        return ParticleBlock(ev, ens, n, psize, std::move(data));
    }

    std::span<const char> particle(size_t i) const {
        if (i >= npart) throw std::out_of_range("Particle index out of range");
        return { particles.data() + i * particle_size, particle_size };
    }
};


// Accessor base class
class Accessor {
public:
    virtual void on_particle_block(const ParticleBlock& block) {}
    virtual void on_end_block(const EndBlock& block) {}
    virtual ~Accessor() = default;

    void set_layout(const std::unordered_map<Quantity, size_t>* layout_in);

    template<typename T>
    T quantity(const std::string& name, const ParticleBlock& block, size_t particle_index) const;

    int32_t get_int(const std::string& name, const ParticleBlock& block, size_t i) const;
    double get_double(const std::string& name, const ParticleBlock& block, size_t i) const;
    virtual void on_header(Header& header_in){};
protected:
    const std::unordered_map<Quantity, size_t>* layout = nullptr;
  std::optional<Header> header = std::nullopt;

};


template<typename T>
T Accessor::quantity(const std::string& name, const ParticleBlock& block, size_t particle_index) const {
    if (!layout) {
        throw std::runtime_error("Layout not set in Accessor");
    }
    if (particle_index >= block.particles.size()) {
        throw std::out_of_range("Invalid particle index");
    }
    return get_quantity<T>(block.particle(particle_index), name, *layout);
}

// BinaryReader class
class BinaryReader {
public:
    BinaryReader(const std::string& filename,
                 const std::vector<std::string>& selected,
                 std::shared_ptr<Accessor> accessor_in);
    void read();

private:
    std::ifstream file;
    size_t particle_size = 0;
    std::shared_ptr<Accessor> accessor;
    std::unordered_map<Quantity, size_t> layout;

    bool check_next(std::ifstream& bfile);
};

#endif // BINARY_READER_H
