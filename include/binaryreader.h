#ifndef BINARY_READER_H
#define BINARY_READER_H
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>

#include "block.h"
#include "particles.h"
#include "quantities.h"
namespace brass {

template <class T>
inline T extract_and_advance(const std::vector<std::byte>& buf, size_t& off) {
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
    Header header_;
    std::ifstream file_;
    std::size_t particle_blocks_read_ = 0;
    std::size_t end_blocks_read_ = 0;
    std::size_t interaction_blocks_read_ = 0;
    std::vector<std::string> quantities_;
    const bool skip_elastic_;

    std::shared_ptr<const ParticleLayout> layout_;
    Header read_header();
    template <class T>
    T read_pod() {
        static_assert(std::is_trivially_copyable_v<T>);

        T value{};
        file_.read(reinterpret_cast<char*>(&value), sizeof(T));

        if (!file_) {
            throw std::runtime_error("read_pod: failed");
        }

        return value;
    }

    std::vector<std::byte> read_chunk(std::size_t n);
};
}  // namespace brass

#endif  // BINARY_READER_H
