#include "binaryreader.h"

#include <stdexcept>

namespace brass {
BinaryReader::BinaryReader(const std::string& filename,
                           std::vector<std::string> quantities,
                           bool skip_elastic)
    : file_(filename, std::ios::binary),
      layout_(std::make_shared<ParticleLayout>(
          particle_layout_from_quantities(quantities))),
      skip_elastic_(skip_elastic) {
    if (!file_) {
        throw std::runtime_error("BinaryReader: could not open file");
    }

    header_ = read_header();
}

Header BinaryReader::read_header() {
    Header h;

    file_.read(h.magic_number.data(), 4);
    if (!file_) {
        throw std::runtime_error("Header: failed to read magic number");
    }

    h.magic_number[4] = '\0';

    h.format_version = read_pod<uint16_t>();
    h.format_variant = read_pod<uint16_t>();

    const uint32_t len = read_pod<uint32_t>();

    if (len > 0) {
        std::string version(len, '\0');

        file_.read(version.data(), static_cast<std::streamsize>(len));
        if (!file_) {
            throw std::runtime_error("Header: failed to read SMASH version");
        }

        h.smash_version = std::move(version);
    }

    return h;
}

std::vector<std::byte> BinaryReader::read_chunk(std::size_t size) {
    std::vector<std::byte> buf(size);

    if (size > 0) {
        file_.read(reinterpret_cast<char*>(buf.data()),
                   static_cast<std::streamsize>(size));

        if (!file_) {
            throw std::runtime_error("read_chunk: failed");
        }
    }

    return buf;
}
std::optional<Block> BinaryReader::read() {
    while (true) {
        char tag{};
        if (!file_.read(&tag, 1)) {
            return std::nullopt;
        }

        switch (tag) {
            case 'p': {
                ParticleBlock block;
                block.event_number = read_pod<int32_t>();
                block.ensemble_number = read_pod<int32_t>();

                const uint32_t npart = read_pod<uint32_t>();

                auto bytes = read_chunk(static_cast<std::size_t>(npart) *
                                        layout_->particle_size);

                if (skip_elastic_ && npart == 2) {
                    continue;
                }

                block.particles = Particles(std::move(bytes), layout_);

                ++particle_blocks_read_;
                return block;
            }

            case 'f': {
                EndBlock block;
                block.event_number = read_pod<uint32_t>();
                block.ensemble_number = read_pod<int32_t>();
                block.impact_parameter = read_pod<double>();

                const char empty = read_pod<char>();
                block.empty = empty != 0;

                ++end_blocks_read_;
                return block;
            }

            case 'i': {
                throw std::runtime_error(
                    "InteractionBlock reading not implemented yet");
            }

            default:
                throw std::runtime_error("BinaryReader: unknown block tag");
        }
    }
}

}  // namespace brass
