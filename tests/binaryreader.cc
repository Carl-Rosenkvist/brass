#include "binaryreader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "doctest.h"

namespace fs = std::filesystem;

template <class T>
static void write(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("write failed");
    }
}

static void write_bytes(std::ofstream& out, const char* data,
                        std::size_t size) {
    out.write(data, static_cast<std::streamsize>(size));
    if (!out) {
        throw std::runtime_error("write failed");
    }
}

static void write_header(std::ofstream& out) {
    write_bytes(out, "SMSH", 4);

    write<uint16_t>(out, 9);
    write<uint16_t>(out, 1);

    const std::string version = "SMASH-3.1";

    write<uint32_t>(out, static_cast<uint32_t>(version.size()));
    write_bytes(out, version.data(), version.size());
}

static void write_particle_block(std::ofstream& out) {
    write<char>(out, 'p');

    write<int32_t>(out, 42);  // event
    write<int32_t>(out, 7);   // ensemble
    write<uint32_t>(out, 2);  // npart

    // quantities = {"px", "pdg"}
    write<double>(out, 1.25);
    write<int32_t>(out, 211);

    write<double>(out, -3.5);
    write<int32_t>(out, -211);
}

static void write_end_block(std::ofstream& out) {
    write<char>(out, 'f');

    write<uint32_t>(out, 42);
    write<int32_t>(out, 7);
    write<double>(out, 1.5);
    write<char>(out, 'x');
}

TEST_CASE("BinaryReader reads header, particle block, and end block") {
    const fs::path tmp = fs::temp_directory_path() / "br_test_px_pdg.bin";

    {
        std::ofstream out(tmp, std::ios::binary);
        REQUIRE(out.is_open());

        write_header(out);
        write_particle_block(out);
        write_end_block(out);
    }

    brass::BinaryReader reader(tmp.string(), {"px", "pdg"});

    const auto& header = reader.header();

    CHECK(header.format_version == 9);
    CHECK(header.format_variant == 1);
    CHECK(header.smash_version == "SMASH-3.1");

    auto first = reader.read();

    REQUIRE(first.has_value());
    REQUIRE(std::holds_alternative<brass::ParticleBlock>(*first));

    const auto& block = std::get<brass::ParticleBlock>(*first);
    const auto& particles = block.particles;

    CHECK(block.event_number == 42);
    CHECK(block.ensemble_number == 7);

    CHECK(particles.size() == 2);
    CHECK_FALSE(particles.empty());
    CHECK(particles.particle_size() == sizeof(double) + sizeof(int32_t));

    const auto& px = particles.column<double>("px");
    const auto& pdg = particles.column<int32_t>("pdg");

    REQUIRE(px.size() == 2);
    REQUIRE(pdg.size() == 2);

    CHECK(px[0] == doctest::Approx(1.25));
    CHECK(px[1] == doctest::Approx(-3.5));

    CHECK(pdg[0] == 211);
    CHECK(pdg[1] == -211);

    auto second = reader.read();

    REQUIRE(second.has_value());
    REQUIRE(std::holds_alternative<brass::EndBlock>(*second));

    const auto& end = std::get<brass::EndBlock>(*second);

    CHECK(end.event_number == 42u);
    CHECK(end.ensemble_number == 7);
    CHECK(end.impact_parameter == doctest::Approx(1.5));
    CHECK(end.empty);

    CHECK_FALSE(reader.read().has_value());

    fs::remove(tmp);
}
