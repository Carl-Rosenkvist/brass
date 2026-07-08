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
                constexpr std::size_t HDR =
                    sizeof(int32_t) + sizeof(int32_t) + sizeof(uint32_t);

                auto header = read_chunk(HDR);
                std::size_t off = 0;

                ParticleBlock block;
                block.event_number = extract_and_advance<int32_t>(header, off);
                block.ensemble_number =
                    extract_and_advance<int32_t>(header, off);

                const uint32_t npart =
                    extract_and_advance<uint32_t>(header, off);

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
                constexpr std::size_t HDR = sizeof(uint32_t) + sizeof(int32_t) +
                                            sizeof(double) + sizeof(char);

                auto header = read_chunk(HDR);
                std::size_t off = 0;

                EndBlock block;
                block.event_number = extract_and_advance<uint32_t>(header, off);
                block.ensemble_number =
                    extract_and_advance<int32_t>(header, off);
                block.impact_parameter =
                    extract_and_advance<double>(header, off);

                const char empty = extract_and_advance<char>(header, off);
                block.empty = empty != 0;

                ++end_blocks_read_;
                return block;
            }

            case 'i': {
                constexpr std::size_t HDR = sizeof(int32_t) + sizeof(int32_t) +
                                            3 * sizeof(double) +
                                            sizeof(int32_t);

                auto header = read_chunk(HDR);
                std::size_t off = 0;

                InteractionBlock block;
                block.n_in = extract_and_advance<int32_t>(header, off);
                block.n_out = extract_and_advance<int32_t>(header, off);
                block.rho = extract_and_advance<double>(header, off);
                block.sigma = extract_and_advance<double>(header, off);
                block.sigma_p = extract_and_advance<double>(header, off);
                block.process = extract_and_advance<int32_t>(header, off);
                const auto n_in = static_cast<std::size_t>(block.n_in);
                const auto n_out = static_cast<std::size_t>(block.n_out);

                auto bytes =
                    read_chunk((n_in + n_out) * layout_->particle_size);

                const auto incoming_size = n_in * layout_->particle_size;

                auto split = bytes.begin() + incoming_size;

                std::vector<std::byte> incoming_bytes(bytes.begin(), split);
                std::vector<std::byte> outgoing_bytes(split, bytes.end());
                block.incoming = Particles(std::move(incoming_bytes), layout_);
                block.outgoing = Particles(std::move(outgoing_bytes), layout_);

                ++interaction_blocks_read_;
                return block;
            }

            default:
                throw std::runtime_error("BinaryReader: unknown block tag");
        }
    }
}
}  // namespace brass
