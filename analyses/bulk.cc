#include <cmath>
#include <unordered_map>

#include "accessor.h"
#include "analysis.h"
#include "analysisregister.h"
#include "histogram2d.h"
namespace py = pybind11;

class BulkObservables : public Analysis {
   public:
    explicit BulkObservables(const std::string& name)
        : Analysis(name),
          y_min_(-4.0),
          y_max_(4.0),
          y_bins_(30),
          pt_min_(0.0),
          pt_max_(3.0),
          pt_bins_(30),
          n_events_(0) {}

    void analyze_particle_block(const ParticleBlock& block,
                                const Accessor& accessor) override {
        ++n_events_;

        init_handles_once(accessor);

        for (uint32_t i = 0; i < block.npart; ++i) {
            const double E = accessor.get_double_fast(block, h_p0_, i);
            const double pz = accessor.get_double_fast(block, h_pz_, i);

            if (E <= std::abs(pz)) continue;

            const double px = accessor.get_double_fast(block, h_px_, i);
            const double py = accessor.get_double_fast(block, h_py_, i);
            const int pdg = accessor.get_int_fast(block, h_pdg_, i);

            const double y = 0.5 * std::log((E + pz) / (E - pz));
            const double pt = std::sqrt(px * px + py * py);

            const int y_bin = int((y - y_min_) * inv_dy_);
            const int pt_bin = int((pt - pt_min_) * inv_dpt_);

            if ((unsigned)y_bin < y_bins_ && (unsigned)pt_bin < pt_bins_) {
                obs_for(pdg).d2N_dpT_dy.fill(pt, y);
            }
        }
    }

    py::dict finalize(py::dict results) override {
        for (auto item : results) {
            py::dict meta = py::cast<py::dict>(item.second);
            py::str key(name());

            if (!meta.contains(key)) continue;

            py::dict state = meta[key].cast<py::dict>();
            if (!state.contains("n_events")) continue;

            const double n_events = state["n_events"].cast<double>();
            if (n_events == 0.0) continue;

            const double dy = (y_max_ - y_min_) / double(y_bins_);
            const double dpt = (pt_max_ - pt_min_) / double(pt_bins_);
            const double norm = n_events * dy * dpt;

            py::dict spectra = state["spectra"].cast<py::dict>();
            for (auto s : spectra) {
                normalize_hist_dict(s.second, norm);
            }

            meta[key] = state;
        }

        return results;
    }

    py::dict to_state_dict() const override {
        py::dict state;

        // --- Scalars ---
        state["n_events"] = n_events_;

        // --- Metadata (non-mergeable) ---
        py::dict meta;

        py::list pt_edges;
        for (size_t i = 0; i <= pt_bins_; ++i) {
            pt_edges.append(pt_min_ +
                            double(i) * (pt_max_ - pt_min_) / double(pt_bins_));
        }

        py::list y_edges;
        for (size_t i = 0; i <= y_bins_; ++i) {
            y_edges.append(y_min_ +
                           double(i) * (y_max_ - y_min_) / double(y_bins_));
        }

        meta["pt_edges"] = pt_edges;
        meta["y_edges"] = y_edges;

        state["meta"] = meta;

        // --- Spectra ---
        py::dict spectra;
        for (const auto& [pdg, obs] : per_pdg_) {
            spectra[py::int_(pdg)] = hist2d_to_state_dict(obs.d2N_dpT_dy);
        }
        state["spectra"] = spectra;

        return state;
    }

    void save(py::dict, const std::string&) override {}

   private:
    // ---------- Helpers ----------

    void init_handles_once(const Accessor& accessor) {
        static thread_local bool initialized = false;
        if (initialized) return;

        h_p0_ = accessor.resolve("p0");
        h_pz_ = accessor.resolve("pz");
        h_px_ = accessor.resolve("px");
        h_py_ = accessor.resolve("py");
        h_pdg_ = accessor.resolve("pdg");

        inv_dy_ = 1.0 / ((y_max_ - y_min_) / double(y_bins_));
        inv_dpt_ = 1.0 / ((pt_max_ - pt_min_) / double(pt_bins_));

        initialized = true;
    }

    static void normalize_hist_dict(py::handle h, double norm) {
        py::dict d = py::cast<py::dict>(h);
        py::list counts = d["counts"].cast<py::list>();
        for (py::ssize_t i = 0; i < py::len(counts); ++i) {
            counts[i] = counts[i].cast<double>() / norm;
        }
    }

    struct Obs {
        Histogram2D d2N_dpT_dy;
        Obs(double pt_min, double pt_max, size_t pt_bins, double y_min,
            double y_max, size_t y_bins)
            : d2N_dpT_dy(pt_min, pt_max, pt_bins, y_min, y_max, y_bins) {}
    };

    Obs& obs_for(int pdg) {
        auto [it, _] = per_pdg_.try_emplace(pdg, pt_min_, pt_max_, pt_bins_,
                                            y_min_, y_max_, y_bins_);
        return it->second;
    }

   private:
    // ---------- Configuration ----------
    double y_min_, y_max_;
    double pt_min_, pt_max_;
    size_t y_bins_, pt_bins_;

    // ---------- Cached binning ----------
    double inv_dy_{0.0};
    double inv_dpt_{0.0};

    // ---------- Histograms ----------
    std::unordered_map<int, Obs> per_pdg_;

    // ---------- State ----------
    int n_events_;

    // ---------- Accessor handles ----------
    Accessor::QuantityHandle h_p0_, h_pz_, h_px_, h_py_, h_pdg_;
};

REGISTER_ANALYSIS("bulk", BulkObservables);
