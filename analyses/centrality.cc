#include <cmath>

#include "accessor.h"
#include "analysis.h"
#include "analysisregister.h"
#include "histogram1d.h"

namespace py = pybind11;

class CentralityObservables : public Analysis {
   public:
    explicit CentralityObservables(const std::string& name)
        : Analysis(name),
          eta_min_(-6.0),
          eta_max_(6.0),
          eta_bins_(120),
          dNch_deta_(eta_min_, eta_max_, eta_bins_),
          n_events_(0),
          n_spectators_(0),
          n_charged_total_(0) {}

    void analyze_particle_block(const ParticleBlock& block,
                                const Accessor& accessor) override {
        ++n_events_;
        init_handles_once(accessor);

        for (uint32_t i = 0; i < block.npart; ++i) {
            const int pdg = accessor.get_int_fast(block, h_pdg_, i);

            // ---- spectator nucleons (ncoll == 0) ----
            if ((pdg == 2212 || pdg == 2112) &&
                accessor.get_int_fast(block, h_ncoll_, i) == 0) {
                ++n_spectators_;
            }

            // ---- charged particles ----
            const int charge = accessor.get_int_fast(block, h_charge_, i);
            if (charge == 0) continue;

            ++n_charged_total_;

            const double px = accessor.get_double_fast(block, h_px_, i);
            const double py = accessor.get_double_fast(block, h_py_, i);
            const double pz = accessor.get_double_fast(block, h_pz_, i);

            const double p = std::sqrt(px * px + py * py + pz * pz);
            if (p <= std::abs(pz)) continue;

            const double eta = 0.5 * std::log((p + pz) / (p - pz));
            dNch_deta_.fill(eta);
        }
    }

    py::dict to_state_dict() const override {
        py::dict state;

        // ---- scalars ----
        state["n_events"] = n_events_;
        state["n_spectators"] = n_spectators_;
        state["n_charged_total"] = n_charged_total_;

        // ---- histogram ----
        state["dNch_deta"] = hist1d_to_state_dict(dNch_deta_);

        // ---- metadata ----
        py::dict meta;
        py::list eta_edges;
        for (size_t i = 0; i <= eta_bins_; ++i) {
            eta_edges.append(eta_min_ + double(i) * (eta_max_ - eta_min_) /
                                            double(eta_bins_));
        }
        meta["eta_edges"] = eta_edges;
        state["meta"] = meta;

        return state;
    }

    py::dict finalize(py::dict results) override {
        for (auto item : results) {
            py::dict meta = py::cast<py::dict>(item.second);
            py::str key(name());
            if (!meta.contains(key)) continue;

            py::dict state = meta[key].cast<py::dict>();
            const double n_ev = state["n_events"].cast<double>();
            if (n_ev == 0.0) continue;

            const double deta = (eta_max_ - eta_min_) / double(eta_bins_);

            py::dict h = state["dNch_deta"].cast<py::dict>();

            // ---- get numpy array, NOT list ----
            py::array counts = h["counts"].cast<py::array>();

            auto buf = counts.mutable_unchecked<double, 1>();
            const ssize_t n = buf.shape(0);

            for (ssize_t i = 0; i < n; ++i) {
                buf(i) /= (n_ev * deta);
            }

            // Optional convenience
            state["mean_n_charged"] =
                state["n_charged_total"].cast<double>() / n_ev;

            meta[key] = state;
        }
        return results;
    }

    void save(py::dict, const std::string&) override {}

   private:
    // ---------- helpers ----------

    void init_handles_once(const Accessor& accessor) {
        static thread_local bool init = false;
        if (init) return;

        h_pdg_ = accessor.resolve("pdg");
        h_px_ = accessor.resolve("px");
        h_py_ = accessor.resolve("py");
        h_pz_ = accessor.resolve("pz");
        h_ncoll_ = accessor.resolve("ncoll");
        h_charge_ = accessor.resolve("charge");

        init = true;
    }

   private:
    // ---------- configuration ----------
    double eta_min_, eta_max_;
    size_t eta_bins_;

    // ---------- histograms ----------
    Histogram1D dNch_deta_;

    // ---------- state ----------
    int n_events_;
    int n_spectators_;
    int n_charged_total_;

    // ---------- accessors ----------
    Accessor::QuantityHandle h_pdg_, h_px_, h_py_, h_pz_, h_ncoll_, h_charge_;
};

REGISTER_ANALYSIS("centrality", CentralityObservables);
