#include "blocks.h"

#include <iostream>

Header Header::read_from(std::ifstream& bfile) {
    Header h;
    bfile.read(h.magic_number.data(), 4);
    h.magic_number[4] = '\0';

    h.format_version = read_pod<uint16_t>(bfile);
    h.format_variant = read_pod<uint16_t>(bfile);
    uint32_t len = read_pod<uint32_t>(bfile);
    if (!bfile) throw std::runtime_error("Header: failed to read fields");

    if (len) {
        auto tmp = read_chunk(bfile, len);
        h.smash_version.assign(tmp.begin(), tmp.end());
    }
    return h;
}

void Header::print() const {
    std::cout << "Magic Number:   " << magic_number.data() << "\n"
              << "Format Version: " << format_version << "\n"
              << "Format Variant: " << format_variant << "\n"
              << "Smash Version:  " << smash_version << "\n";
}

EndBlock EndBlock::read_from(std::ifstream& bfile, const Format& /*fmt*/) {
    EndBlock e;
    auto buf = read_chunk(bfile, EndBlock::SIZE);
    size_t off = 0;
    e.event_number = extract_and_advance<uint32_t>(buf, off);
    e.ensamble_number = extract_and_advance<uint32_t>(buf, off);
    e.impact_parameter = extract_and_advance<double>(buf, off);
    uint8_t raw = extract_and_advance<uint8_t>(buf, off);
    e.empty = (raw != 0);
    return e;
}

ParticleBlock ParticleBlock::read_from(std::ifstream& bfile, size_t psize,
                                       const Format& /*fmt*/) {
    ParticleBlock p;
    p.particle_size = psize;

    constexpr size_t HDR = sizeof(int32_t) + sizeof(int32_t) + sizeof(uint32_t);
    auto header = read_chunk(bfile, HDR);
    size_t off = 0;

    p.event_number = extract_and_advance<int32_t>(header, off);
    p.ensamble_number = extract_and_advance<int32_t>(header, off);
    p.npart = extract_and_advance<uint32_t>(header, off);

    const size_t bytes =
        checked_mul(static_cast<size_t>(p.npart), psize, "ParticleBlock bytes");
    p.particles = read_chunk(bfile, bytes);
    return p;
}

InteractionBlock InteractionBlock::read_from(std::ifstream& bfile, size_t psize,
                                             const Format& fmt) {
    InteractionBlock ib;
    ib.particle_size = psize;

    // header bytes in one go (adjust size if fields change with fmt)
    constexpr size_t HDR = sizeof(int32_t) + sizeof(int32_t) +
                           3 * sizeof(double) + sizeof(int32_t);
    auto header = read_chunk(bfile, HDR);
    size_t off = 0;

    ib.n_in = extract_and_advance<int32_t>(header, off);
    ib.n_out = extract_and_advance<int32_t>(header, off);
    ib.rho = extract_and_advance<double>(header, off);
    ib.sigma = extract_and_advance<double>(header, off);
    ib.sigma_p = extract_and_advance<double>(header, off);
    ib.process = extract_and_advance<int32_t>(header, off);

    const size_t nin_bytes =
        checked_mul(ib.n_in_sz(), psize, "Interaction incoming");
    const size_t nout_bytes =
        checked_mul(ib.n_out_sz(), psize, "Interaction outgoing");

    ib.incoming = read_chunk(bfile, nin_bytes);
    ib.outgoing = read_chunk(bfile, nout_bytes);
    return ib;
}
