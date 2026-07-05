#ifndef BINARY_READER_H
#define BINARY_READER_H
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>

#include "particles.h"
#include "quantities.h"
namespace brass {

struct Header {
    std::array<char, 5> magic_number{{0, 0, 0, 0, 0}};  // includes NUL
    uint16_t format_version = 0;
    uint16_t format_variant = 0;
    std::string smash_version;
};

struct ParticleBlock {
    int32_t event_number = 0;
    int32_t ensemble_number = 0;
    Particles particles;
};

struct EndBlock {
    uint32_t event_number = 0;
    int32_t ensemble_number = 0;
    double impact_parameter = 0.0;
    bool empty = false;
};

struct InteractionBlock {
    int32_t n_in = 0;
    int32_t n_out = 0;
    double rho = 0.0;
    double sigma = 0.0;
    double sigma_p = 0.0;
    int32_t process = 0;

    Particles incoming;
    Particles outgoing;
};

using Block = std::variant<ParticleBlock, EndBlock, InteractionBlock>;

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
