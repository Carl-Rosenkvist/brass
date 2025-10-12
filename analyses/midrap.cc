#include "analysis.h"
#include "analysisregister.h"   // <- fix name
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <fstream>
#include <mutex>
#include <atomic>
#include <cmath>
#include "mergekey.h"            // <- ensure label_from_keyset is declared

class Midrapidity : public Analysis {
public:
  explicit Midrapidity(const std::string& name)      // <- match macro
  : Analysis(name),
    n_events_(0),
    pdgs_({211,-211,321,-321}),
    pdg_set_(pdgs_.begin(), pdgs_.end()) {}

  Analysis& operator+=(const Analysis& other) override {
    auto const* o = dynamic_cast<const Midrapidity*>(&other);
    if (!o) throw std::runtime_error("merge mismatch: Midrapidity");
    n_events_ += o->n_events_;
    for (auto const& [pdg, cnt] : o->pdg_count_) pdg_count_[pdg] += cnt;
    return *this;
  }

  void analyze_particle_block(const ParticleBlock& b, const Accessor& a) override {
    ++n_events_;
    for (size_t i = 0; i < b.npart; ++i) {
      const int pdg = a.get_int("pdg", b, i);
      if (!pdg_set_.empty() && !pdg_set_.count(pdg)) continue;

      const double E  = a.get_double("p0", b, i);
      const double pz = a.get_double("pz", b, i);
      if (E <= std::abs(pz)) continue;
      const double y = 0.5 * std::log((E + pz) / (E - pz));
      if (std::abs(y) > 0.5) continue;

      pdg_count_[pdg] += 1.0;
    }
  }

  void finalize() override {
    if (n_events_ <= 0) return;
    for (auto& [pdg, cnt] : pdg_count_) cnt /= static_cast<double>(n_events_);
  }

  void save(const std::string& out_dir) override {
    static bool s_header_written = false;
    static std::mutex s_io_mtx;

    const std::string out_path = out_dir + "/mid_rap.csv";
    std::lock_guard<std::mutex> lk(s_io_mtx);

    const bool first = !s_header_written;
    std::ofstream ofs(out_path, first ? std::ios::trunc : std::ios::app);
    if (!ofs) throw std::runtime_error("Midrapidity: cannot open " + out_path);

    if (first) {
      ofs << "label";
      for (int pdg : pdgs_) ofs << "," << pdg;
      ofs << "\n";
      s_header_written = true;
    }

    const std::string meta = label_from_keyset(keys);
    ofs << meta;
    for (int pdg : pdgs_) {
      const auto it = pdg_count_.find(pdg);
      ofs << "," << (it == pdg_count_.end() ? 0.0 : it->second);
    }
    ofs << "\n";
  }

private:
  int n_events_;
  std::vector<int> pdgs_;
  std::unordered_set<int> pdg_set_;
  std::unordered_map<int, double> pdg_count_;
};

REGISTER_ANALYSIS("midrapidity", Midrapidity);
