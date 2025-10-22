#include "analysis.h"
#include "analysisregister.h"   
#include "histogram2d.h"
#include <yaml-cpp/yaml.h>       // <- needed for YAML::Emitter
#include <cmath>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <string>
#include <mutex>                  // <- needed for std::mutex

// If to_yaml is declared elsewhere, include that header too:
// #include "yaml_utils.h"  // or wherever to_yaml(...) lives

class BulkObservables : public Analysis {
public:
    explicit BulkObservables(const std::string& name)   // <- match macro
      : Analysis(name),
        y_min(-4.0), y_max(4.0), y_bins(30),
        pt_min(0.0), pt_max(3.0), pt_bins(30),
        d2N_dpT_dy(pt_min, pt_max, pt_bins, y_min, y_max, y_bins),
        n_events(0) {}

    Analysis& operator+=(const Analysis& other) override {
        auto const* o = dynamic_cast<const BulkObservables*>(&other);
        if (!o) throw std::runtime_error("merge mismatch");
        d2N_dpT_dy += o->d2N_dpT_dy;
        n_events   += o->n_events;
        for (auto const& [pdg, src] : o->per_pdg_) {
            auto& dst = obs_for(pdg);
            dst.d2N_dpT_dy += src.d2N_dpT_dy;
        }
        return *this;
    }


    void analyze_particle_block(const ParticleBlock& b, const Accessor& a) override {
        ++n_events;

        static thread_local bool init = false;
        static thread_local Accessor::QuantityHandle h_p0, h_pz, h_px, h_py, h_pdg;
        static thread_local double inv_dy, inv_dpt;

        if (!init) {
            h_p0  = a.resolve("p0");
            h_pz  = a.resolve("pz");
            h_px  = a.resolve("px");
            h_py  = a.resolve("py");
            h_pdg = a.resolve("pdg");
            inv_dy  = 1.0 / ((y_max - y_min) / double(y_bins));
            inv_dpt = 1.0 / ((pt_max - pt_min) / double(pt_bins));
            init = true;
        }

        for (uint32_t i = 0; i < b.npart; ++i) {
            const double E  = a.get_double_fast(b, h_p0,  i);
            const double pz = a.get_double_fast(b, h_pz,  i);
            if (E <= std::abs(pz)) continue;

            const double px  = a.get_double_fast(b, h_px,  i);
            const double py  = a.get_double_fast(b, h_py,  i);
            const int    pdg = a.get_int_fast   (b, h_pdg, i);

            const double y  = 0.5 * std::log((E + pz) / (E - pz));
            const double pt = std::sqrt(px*px + py*py);

            const int by = int((y  - y_min) * inv_dy);
            const int bp = int((pt - pt_min) * inv_dpt);
            if ((unsigned)by < y_bins && (unsigned)bp < pt_bins) {
                d2N_dpT_dy.fill(pt, y);
                obs_for(pdg).d2N_dpT_dy.fill(pt, y);
                // or add_bin(bp,by) if you implement a direct-bin increment
            }
        }
    }
    void finalize() override {
        if (n_events == 0) return;
        const double dy   = (y_max - y_min) / static_cast<double>(y_bins);
        const double dpt  = (pt_max - pt_min) / static_cast<double>(pt_bins);
        const double norm = static_cast<double>(n_events) * dy * dpt;
        d2N_dpT_dy.scale(1.0 / norm);
        for (auto& [_, o] : per_pdg_) o.d2N_dpT_dy.scale(1.0 / norm);
    }

    void save(const std::string& dir) override {
        static bool s_first = true;
        static std::mutex s_io_mtx;
        const std::string out_path = dir + "/bulk.yaml";
        std::lock_guard<std::mutex> lk(s_io_mtx);

        std::ofstream f(out_path, s_first ? std::ios::trunc : std::ios::app);
        if (!f) throw std::runtime_error("BulkObservables: cannot open " + out_path);

        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "merge_key"     << YAML::Value; to_yaml(out, keys);
        out << YAML::Key << "n_events"      << YAML::Value << n_events;
        out << YAML::Key << "smash_version" << YAML::Value << get_smash_version();

        // Optionally also emit the inclusive spectrum:
        out << YAML::Key << "inclusive" << YAML::Value;
        to_yaml(out, "pt", "y", d2N_dpT_dy);

        out << YAML::Key << "spectra" << YAML::Value << YAML::BeginMap;
        for (auto const& [pdg, o] : per_pdg_) {
            out << YAML::Key << std::to_string(pdg) << YAML::Value;
            to_yaml(out, "pt", "y", o.d2N_dpT_dy);
        }
        out << YAML::EndMap; // spectra
        out << YAML::EndMap; // doc
        f << "---\n" << out.c_str() << "\n";
        s_first = false;
    }

private:
    struct Obs {
        Histogram2D d2N_dpT_dy;
        Obs(double pt_min, double pt_max, size_t pt_bins,
            double y_min, double y_max, size_t y_bins)
          : d2N_dpT_dy(pt_min, pt_max, pt_bins, y_min, y_max, y_bins) {}
    };

    Obs& obs_for(int pdg) {
        auto [it, _] = per_pdg_.try_emplace(
            pdg, pt_min, pt_max, pt_bins, y_min, y_max, y_bins);
        return it->second;
    }

    double y_min, y_max, pt_min, pt_max;
    size_t y_bins, pt_bins;
    Histogram2D d2N_dpT_dy;
    int n_events;
    std::unordered_map<int, Obs> per_pdg_;
};

REGISTER_ANALYSIS("bulk", BulkObservables);
