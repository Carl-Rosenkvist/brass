#ifndef BINARY_READER_H
#define BINARY_READER_H
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "block.h"
#include "particles.h"
#include "quantities.h"
namespace brass {

template <class T, class Buffer>
inline T extract_and_advance(const Buffer& buf, size_t& off) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "extract: T must be trivially copyable");
    if (off + sizeof(T) > buf.size())
        throw std::runtime_error("extract: out of bounds");
    T v;
    std::memcpy(&v, buf.data() + off, sizeof(T));
    off += sizeof(T);
    return v;
}

class BinaryReader {
   public:
    explicit BinaryReader(const std::string& filename,
                          std::vector<std::string> quantities,
                          bool skip_elastic = false);

    const Header& header() const { return header_; }
    std::optional<Block> read();
    std::size_t particle_blocks_read() const { return particle_blocks_read_; }
    std::size_t end_blocks_read() const { return end_blocks_read_; }
    std::size_t interaction_blocks_read() const {
        return interaction_blocks_read_;
    }

   private:
    struct FileCloser {
        void operator()(std::FILE* f) const { std::fclose(f); }
    };

    // C stdio with a large buffer: much faster than std::ifstream for the
    // many block-sized reads of a SMASH file.
    std::unique_ptr<std::FILE, FileCloser> file_;
    std::vector<char> file_buffer_;

    Header header_;
    std::size_t particle_blocks_read_ = 0;
    std::size_t end_blocks_read_ = 0;
    std::size_t interaction_blocks_read_ = 0;
    const bool skip_elastic_;

    std::shared_ptr<const ParticleLayout> layout_;
    Header read_header();

    // Read exactly n bytes into dst; false at end of file or on error.
    bool read_bytes(void* dst, std::size_t n) {
        return n == 0 || std::fread(dst, 1, n, file_.get()) == n;
    }

    template <class T>
    T read_pod() {
        static_assert(std::is_trivially_copyable_v<T>);

        T value{};
        if (!read_bytes(&value, sizeof(T))) {
            throw std::runtime_error("read_pod: failed");
        }

        return value;
    }

    // Fixed-size block header, read onto the stack.
    template <std::size_t N>
    std::array<std::byte, N> read_array() {
        std::array<std::byte, N> buf;
        if (!read_bytes(buf.data(), N)) {
            throw std::runtime_error("read_chunk: failed");
        }
        return buf;
    }

    ByteBuffer read_chunk(std::size_t n);
};
}  // namespace brass

#endif  // BINARY_READER_H
