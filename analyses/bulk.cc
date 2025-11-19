#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <unordered_map>

#include "analysis.h"
#include "analysisregister.h"
#include "histogram2d.h"

namespace py = pybind11;

class BulkObservables : public Analysis {
   public:
    explicit BulkObservables(const std::string& name)
        : Analysis(name),
          y_min(-4.0),
          y_max(4.0),
          y_bins(30),
          pt_min(0.0),
          pt_max(3.0),
          pt_bins(30),
          d2N_dpT_dy(pt_min, pt_max, pt_bins, y_min, y_max, y_bins),
          n_events(0) {}

    void analyze_particle_block(const ParticleBlock& b,
                                const Accessor& a) override {
        ++n_events;

        static thread_local bool init = false;
        static thread_local Accessor::QuantityHandle h_p0, h_pz, h_px, h_py,
            h_pdg;
        static thread_local double inv_dy, inv_dpt;

        if (!init) {
            h_p0 = a.resolve("p0");
            h_pz = a.resolve("pz");
            h_px = a.resolve("px");
            h_py = a.resolve("py");
            h_pdg = a.resolve("pdg");
            inv_dy = 1.0 / ((y_max - y_min) / double(y_bins));
            inv_dpt = 1.0 / ((pt_max - pt_min) / double(pt_bins));
            init = true;
        }

        for (uint32_t i = 0; i < b.npart; ++i) {
            const double E = a.get_double_fast(b, h_p0, i);
            const double pz = a.get_double_fast(b, h_pz, i);
            if (E <= std::abs(pz)) continue;

            const double px = a.get_double_fast(b, h_px, i);
            const double py = a.get_double_fast(b, h_py, i);
            const int pdg = a.get_int_fast(b, h_pdg, i);

            const double y = 0.5 * std::log((E + pz) / (E - pz));
            const double pt = std::sqrt(px * px + py * py);

            const int by = int((y - y_min) * inv_dy);
            const int bp = int((pt - pt_min) * inv_dpt);
            if ((unsigned)by < y_bins && (unsigned)bp < pt_bins) {
                d2N_dpT_dy.fill(pt, y);
                obs_for(pdg).d2N_dpT_dy.fill(pt, y);
            }
        }
    }

    py::dict finalize(py::dict results) override {
        for (auto item : results) {
            py::handle meta_key = item.first;
            py::dict meta_dict = py::cast<py::dict>(item.second);

            py::str self_name(name());
            if (!meta_dict.contains(self_name)) continue;

            py::dict state = meta_dict[self_name].cast<py::dict>();
            if (!state.contains("n_events")) continue;

            double n_events_total = state["n_events"].cast<double>();
            if (n_events_total == 0.0) {
                meta_dict[self_name] = state;
                continue;
            }

            double dy = (y_max - y_min) / double(y_bins);
            double dpt = (pt_max - pt_min) / double(pt_bins);
            double norm = n_events_total * dy * dpt;

            py::dict incl = state["inclusive"].cast<py::dict>();
            py::list counts_incl = incl["counts"].cast<py::list>();
            for (py::ssize_t i = 0; i < py::len(counts_incl); ++i) {
                counts_incl[i] = counts_incl[i].cast<double>() / norm;
            }

            py::dict spec = state["spectra"].cast<py::dict>();
            for (auto sitem : spec) {
                py::dict h = py::cast<py::dict>(sitem.second);
                py::list c = h["counts"].cast<py::list>();
                for (py::ssize_t i = 0; i < py::len(c); ++i) {
                    c[i] = c[i].cast<double>() / norm;
                }
            }

            meta_dict[self_name] = state;
        }

        return results;
    }

    void save(py::dict, const std::string&) override {}

    py::dict to_state_dict() const override {
        py::dict d;
        d["n_events"] = n_events;
        d["inclusive"] = hist2d_to_state_dict(d2N_dpT_dy);

        py::dict spec;
        for (auto const& p : per_pdg_) {
            spec[py::int_(p.first)] = hist2d_to_state_dict(p.second.d2N_dpT_dy);
        }
        d["spectra"] = spec;

        return d;
    }

   private:
    struct Obs {
        Histogram2D d2N_dpT_dy;
        Obs(double pt_min, double pt_max, size_t pt_bins, double y_min,
            double y_max, size_t y_bins)
            : d2N_dpT_dy(pt_min, pt_max, pt_bins, y_min, y_max, y_bins) {}
    };

    Obs& obs_for(int pdg) {
        auto [it, _] = per_pdg_.try_emplace(pdg, pt_min, pt_max, pt_bins, y_min,
                                            y_max, y_bins);
        return it->second;
    }

    double y_min, y_max, pt_min, pt_max;
    size_t y_bins, pt_bins;
    Histogram2D d2N_dpT_dy;
    int n_events;
    std::unordered_map<int, Obs> per_pdg_;
};

REGISTER_ANALYSIS("bulk", BulkObservables);
