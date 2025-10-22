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
#include <type_traits> // for std::is_same_v

// --- Types ---

enum class QuantityType {
    Double,
    Int32
};

size_t type_size(QuantityType t);

extern const std::unordered_map<std::string, QuantityType> quantity_string_map;

// name -> byte offset within a particle record
std::unordered_map<std::string, size_t>
compute_quantity_layout(const std::vector<std::string>& names);

std::vector<char> read_chunk(std::ifstream& bfile, size_t size);

// --- Template helpers ---

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
               const std::unordered_map<std::string, size_t>& layout)
{
    auto it_info = quantity_string_map.find(name);
    if (it_info == quantity_string_map.end())
        throw std::runtime_error("Unknown quantity: " + name);

    // Type check against requested T
    if constexpr (std::is_same_v<T, double>) {
        if (it_info->second != QuantityType::Double)
            throw std::runtime_error("Requested double, but quantity is not double: " + name);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        if (it_info->second != QuantityType::Int32)
            throw std::runtime_error("Requested int32, but quantity is not int32: " + name);
    } else {
        static_assert(!sizeof(T*), "Unsupported T in get_quantity");
    }

    // Look up by NAME (no enum anymore)
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

// --- Header block ---

class Header {
public:
    const std::array<char, 5> magic_number; // includes NUL
    const uint16_t            format_version;
    const uint16_t            format_variant;
    const std::string         smash_version;

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
        if (!bfile) throw std::runtime_error("Failed to read header (length).");

        std::string smash_ver;
        if (len > 0) {
            std::vector<char> buf(len);
            bfile.read(buf.data(), len);
            smash_ver.assign(buf.begin(), buf.end());
        }
        if (!bfile) throw std::runtime_error("Failed to read header contents.");

        return Header(magic_buf, version, variant, std::move(smash_ver));
    }

    void print() const {
        std::cout << "Magic Number:   " << magic_number.data() << "\n"
                  << "Format Version: " << format_version      << "\n"
                  << "Format Variant: " << format_variant      << "\n"
                  << "Smash Version:  " << smash_version       << "\n";
    }

private:
    Header(std::array<char,5> magic,
           uint16_t version,
           uint16_t variant,
           std::string smash_ver)
        : magic_number(magic),
          format_version(version),
          format_variant(variant),
          smash_version(std::move(smash_ver)) {}
};

// --- End block ---

class EndBlock {
public:
    const uint32_t event_number;
    const uint32_t ensamble_number;
    const double   impact_parameter;
    const bool     empty;

    static constexpr size_t SIZE = 4u + 4u + 8u + 1u;

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

// --- Particle block ---

struct ParticleBlock {
    const int32_t  event_number;
    const int32_t  ensamble_number;
    const uint32_t npart;
    const size_t   particle_size;
    const std::vector<char> particles;

    ParticleBlock(int32_t ev, int32_t ens, uint32_t n, size_t psize, std::vector<char> data)
      : event_number(ev),
        ensamble_number(ens),
        npart(n),
        particle_size(psize),
        particles(std::move(data)) {}

    static ParticleBlock read_from(std::ifstream& bfile, size_t psize) {
        constexpr size_t HEADER_SIZE = sizeof(int32_t) + sizeof(int32_t) + sizeof(uint32_t);
        auto header = read_chunk(bfile, HEADER_SIZE);

        size_t off = 0;
        const int32_t  ev  = extract_and_advance<int32_t>(header, off);
        const int32_t  ens = extract_and_advance<int32_t>(header, off);
        const uint32_t n   = extract_and_advance<uint32_t>(header, off);

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

// --- Accessor base class ---


// --- Accessor base class ---

class Accessor {
public:
    virtual void on_particle_block(const ParticleBlock& block) {}
    virtual void on_end_block(const EndBlock& block) {}
    virtual ~Accessor() = default;

    void set_layout(const std::unordered_map<std::string, size_t>* layout_in);

    const std::unordered_map<std::string, size_t>& layout_map() const {
        if (!layout) throw std::runtime_error("Layout not set in Accessor");
        return *layout;
    }

    template<typename T>
    T quantity(const std::string& name, const ParticleBlock& block, size_t particle_index) const;

    int32_t get_int(const std::string& name, const ParticleBlock& block, size_t i) const;
    double  get_double(const std::string& name, const ParticleBlock& block, size_t i) const;
    virtual void on_header(Header& header_in){};

    // Resolve-once handle for hot loops
    struct QuantityHandle {
        size_t        offset;
        QuantityType  type;
    };

    inline QuantityHandle resolve(const std::string& name) const {
        if (!layout) throw std::runtime_error("Layout not set");
        auto it_off = layout->find(name);
        if (it_off == layout->end()) throw std::runtime_error("Unknown quantity in layout: " + name);
        auto it_ty = quantity_string_map.find(name);
        if (it_ty == quantity_string_map.end()) throw std::runtime_error("Unknown quantity type: " + name);
        return { it_off->second, it_ty->second };
    }

    inline double get_double_fast(const ParticleBlock& b, size_t off, size_t i) const noexcept {
        const char* p = b.particles.data() + i * b.particle_size + off;
        return *reinterpret_cast<const double*>(p);
    }

    inline int32_t get_int_fast(const ParticleBlock& b, size_t off, size_t i) const noexcept {
        const char* p = b.particles.data() + i * b.particle_size + off;
        return *reinterpret_cast<const int32_t*>(p);
    }

    // (Optional) type-checked fast getters for debug builds
    inline double get_double_fast(const ParticleBlock& b, const QuantityHandle& h, size_t i) const noexcept {
#ifndef NDEBUG
        if (h.type != QuantityType::Double) throw std::logic_error("get_double_fast: wrong type");
#endif
        return get_double_fast(b, h.offset, i);
    }
    inline int32_t get_int_fast(const ParticleBlock& b, const QuantityHandle& h, size_t i) const noexcept {
#ifndef NDEBUG
        if (h.type != QuantityType::Int32) throw std::logic_error("get_int_fast: wrong type");
#endif
        return get_int_fast(b, h.offset, i);
    }

protected:
    const std::unordered_map<std::string, size_t>* layout = nullptr;
    std::optional<Header> header = std::nullopt;
};



template<typename T>
T Accessor::quantity(const std::string& name, const ParticleBlock& block, size_t particle_index) const {
    if (!layout) throw std::runtime_error("Layout not set in Accessor");
    if (particle_index >= block.npart) // compare against number of particles, not bytes
        throw std::out_of_range("Invalid particle index");
    return get_quantity<T>(block.particle(particle_index), name, *layout);
}

// --- BinaryReader ---

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
    std::unordered_map<std::string, size_t> layout;

    bool check_next(std::ifstream& bfile);
};

#endif // BINARY_READER_H
