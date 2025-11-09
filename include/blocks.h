#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// ======================== Format (future-proof, defaulted now)
// ========================
struct Format {
    uint16_t version = 0;
    uint16_t variant = 0;
};

// ============================= Helpers (templates inline)
// =============================
inline std::vector<char> read_chunk(std::ifstream& f, size_t n) {
    std::vector<char> buf(n);
    if (n) {
        f.read(buf.data(), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("read_chunk: failed");
    }
    return buf;
}

template <class T>
inline T read_pod(std::ifstream& f) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "read_pod: T must be trivially copyable");
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("read_pod: failed");
    return v;
}

template <class T>
inline T extract_and_advance(const std::vector<char>& buf, size_t& off) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "extract: T must be trivially copyable");
    if (off + sizeof(T) > buf.size())
        throw std::runtime_error("extract: out of bounds");
    T v;
    std::memcpy(&v, buf.data() + off, sizeof(T));
    off += sizeof(T);
    return v;
}

inline size_t checked_mul(size_t a, size_t b, const char* what) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error(std::string("overflow: ") + what);
    return a * b;
}

// ================================ Tag for block loop
// ==================================
enum class BlockTag : char { Particle = 'p', End = 'f', Interaction = 'i' };

inline std::optional<BlockTag> read_tag(std::ifstream& f) {
    char c{};
    if (!f.read(&c, 1)) return std::nullopt;  // EOF or error
    switch (c) {
        case 'p':
            return BlockTag::Particle;
        case 'f':
            return BlockTag::End;
        case 'i':
            return BlockTag::Interaction;
        default:
            return std::nullopt;  // unknown tag -> stop gracefully
    }
}

// =================================== Header
// ==========================================
struct Header {
    std::array<char, 5> magic_number{{0, 0, 0, 0, 0}};  // includes NUL
    uint16_t format_version = 0;
    uint16_t format_variant = 0;
    std::string smash_version;

    static Header read_from(std::ifstream& bfile);
    Format as_format() const { return Format{format_version, format_variant}; }
    void print() const;
};

// ================================== EndBlock
// =========================================
struct EndBlock {
    uint32_t event_number = 0;
    uint32_t ensamble_number = 0;
    double impact_parameter = 0.0;
    bool empty = false;

    static constexpr size_t SIZE = 4u + 4u + 8u + 1u;
    static EndBlock read_from(std::ifstream& bfile, const Format& fmt = {});
};

// ================================= ParticleBlock
// =====================================
struct ParticleBlock {
    int32_t event_number = 0;
    int32_t ensamble_number = 0;
    uint32_t npart = 0;
    size_t particle_size = 0;
    std::vector<char> particles;  // raw bytes (npart * particle_size)

    static ParticleBlock read_from(std::ifstream& bfile, size_t psize,
                                   const Format& fmt = {});

    std::span<const char> particle(size_t i) const {
        if (particle_size == 0) throw std::runtime_error("particle_size==0");
        if (i >= npart) throw std::out_of_range("Particle index out of range");
        return {particles.data() + i * particle_size, particle_size};
    }
};

// =============================== InteractionBlock
// ====================================
struct InteractionBlock {
    int32_t n_in = 0;
    int32_t n_out = 0;

    double rho;
    double sigma;
    double sigma_p;

    int32_t process = 0;
    size_t particle_size = 0;
    std::vector<char> incoming;  // n_in * particle_size
    std::vector<char> outgoing;  // n_out * particle_size

    static InteractionBlock read_from(std::ifstream& bfile, size_t psize,
                                      const Format& fmt = {});

    size_t n_in_sz() const { return static_cast<size_t>(n_in < 0 ? 0 : n_in); }
    size_t n_out_sz() const {
        return static_cast<size_t>(n_out < 0 ? 0 : n_out);
    }

    std::span<const char> incoming_particle(size_t i) const {
        if (i >= n_in_sz())
            throw std::out_of_range("Incoming particle index OOB");
        return {incoming.data() + i * particle_size, particle_size};
    }
    std::span<const char> outgoing_particle(size_t i) const {
        if (i >= n_out_sz())
            throw std::out_of_range("Outgoing particle index OOB");
        return {outgoing.data() + i * particle_size, particle_size};
    }
};
