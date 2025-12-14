// tests/binaryreader.cc
#include "binaryreader.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"

namespace fs = std::filesystem;

// ---------- tiny binary writer for the test (px + pdg only) ----------
static void wr_bytes(std::ofstream& out, const void* p, size_t n) {
    out.write(reinterpret_cast<const char*>(p),
              static_cast<std::streamsize>(n));
    if (!out) throw std::runtime_error("write failed");
}

template <typename T>
static void wr(std::ofstream& out, const T& v) {
    wr_bytes(out, &v, sizeof(T));
}

// ---------- header ----------
static void writeHeader_min(std::ofstream& out) {
    const char magic[4] = {'S', 'M', 'S', 'H'};
    wr_bytes(out, magic, 4);

    const uint16_t ver = 9;
    const uint16_t var = 1;
    wr(out, ver);
    wr(out, var);

    const std::string sv = "SMASH-3.1";
    const uint32_t len = static_cast<uint32_t>(sv.size());
    wr(out, len);
    wr_bytes(out, sv.data(), sv.size());
}

// ---------- particle block ----------
// Layout expected by reader:
// 'p' + int32 event + int32 ensemble + uint32 npart
// + [ double px, int32 pdg ] * npart
static void writeParticleBlock_px_pdg(
    std::ofstream& out, int32_t event, int32_t ensemble,
    const std::vector<std::pair<double, int32_t>>& parts) {
    const char tag = 'p';
    wr(out, tag);
    wr(out, event);
    wr(out, ensemble);

    const uint32_t npart = static_cast<uint32_t>(parts.size());
    wr(out, npart);

    for (const auto& [px, pdg] : parts) {
        wr(out, px);
        wr(out, pdg);
    }
}

// ---------- end block ----------
// Layout expected by reader:
// 'f' + uint32 event + int32 ensemble + double b + uint8 empty
static void writeEndBlock_min(std::ofstream& out, uint32_t event,
                              uint32_t ensemble, double b, uint8_t empty_flag) {
    const char tag = 'f';
    wr(out, tag);
    wr(out, event);
    wr(out, ensemble);
    wr(out, b);
    wr(out, empty_flag);
}

// ---------- trailing info tag ----------
static void writeInfoTag(std::ofstream& out) {
    const char tag = 'i';
    wr(out, tag);
}

// ---------- mock accessor ----------
struct MockAccessor : public Accessor {
    bool saw_header = false;
    uint16_t header_version = 0;
    uint16_t header_variant = 0;
    std::string header_smash_version;

    std::vector<double> pxs;
    std::vector<int32_t> pdgs;

    uint32_t end_event = 0;
    uint32_t end_ensemble = 0;
    double end_b = 0.0;
    bool end_empty = false;

    void on_header(Header& h) override {
        saw_header = true;
        header_version = h.format_version;
        header_variant = h.format_variant;
        header_smash_version = h.smash_version;
    }

    void on_particle_block(const ParticleBlock& block) override {
        for (size_t i = 0; i < block.npart; ++i) {
            pxs.push_back(get_double("px", block, i));
            pdgs.push_back(get_int("pdg", block, i));
        }
    }

    void on_end_block(const EndBlock& e) override {
        end_event = e.event_number;
        end_ensemble = e.ensamble_number;
        end_b = e.impact_parameter;
        end_empty = e.empty;
    }
};

// ---------- test ----------
TEST_CASE("BinaryReader reads header, px/pdg particle block, and end block") {
    const fs::path tmp = fs::temp_directory_path() / "br_test_px_pdg.bin";

    {
        std::ofstream out(tmp, std::ios::binary);
        REQUIRE(out.is_open());

        writeHeader_min(out);

        std::vector<std::pair<double, int32_t>> parts = {{1.25, 211},
                                                         {-3.5, -211}};

        writeParticleBlock_px_pdg(out, 42, 7, parts);
        writeEndBlock_min(out, 42u, 7, 1.5, uint8_t{1});
    }

    auto acc = std::make_shared<MockAccessor>();
    std::vector<std::string> selected = {"px", "pdg"};

    BinaryReader reader(tmp.string(), selected, acc);

    CHECK_NOTHROW(reader.read());

    // header
    CHECK(acc->saw_header);
    CHECK(acc->header_version == 9);
    CHECK(acc->header_variant == 1);
    CHECK(acc->header_smash_version == "SMASH-3.1");

    // particles
    REQUIRE(acc->pxs.size() == 2);
    REQUIRE(acc->pdgs.size() == 2);

    CHECK(acc->pxs[0] == doctest::Approx(1.25));
    CHECK(acc->pdgs[0] == 211);
    CHECK(acc->pxs[1] == doctest::Approx(-3.5));
    CHECK(acc->pdgs[1] == -211);

    // end block
    CHECK(acc->end_event == 42u);
    CHECK(acc->end_ensemble == 7);
    CHECK(acc->end_b == doctest::Approx(1.5));
    CHECK(acc->end_empty == true);

    fs::remove(tmp);
}
