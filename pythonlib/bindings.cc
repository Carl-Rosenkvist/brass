#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <optional>
#include <span>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "analysis.h"
#include "analysisregister.h"
#include "binaryreader.h"

namespace py = pybind11;

// -------------------- Utilities --------------------
std::vector<std::string> list_analyses() {
    return AnalysisRegistry::instance().list_registered();
}

// Trampoline to call Python overrides on Accessor
class PyAccessor : public Accessor {
   public:
    using Accessor::Accessor;

    void on_particle_block(const ParticleBlock& block) override {
        PYBIND11_OVERRIDE(void, Accessor, on_particle_block, block);
    }
    void on_end_block(const EndBlock& block) override {
        PYBIND11_OVERRIDE(void, Accessor, on_end_block, block);
    }
    void on_interaction_block(const InteractionBlock& block) override {
        PYBIND11_OVERRIDE(void, Accessor, on_interaction_block, block);
    }
};

class PythonAnalysis : public Analysis {
   public:
    PythonAnalysis(const std::string& name, py::object py_obj, py::dict opts)
        : Analysis(name), obj_(std::move(py_obj)), opts_(std::move(opts)) {}

    void analyze_particle_block(const ParticleBlock& b,
                                const Accessor& a) override {
        py::gil_scoped_acquire gil;
        obj_.attr("on_particle_block")(py::cast(b), py::cast(a), opts_);
    }

    void analyze_end_block(const EndBlock& b, const Accessor& a) override {
        py::gil_scoped_acquire gil;
        obj_.attr("on_end_block")(py::cast(b), py::cast(a), opts_);
    }

    void analyze_interaction_block(const InteractionBlock& b,
                                   const Accessor& a) override {
        py::gil_scoped_acquire gil;
        obj_.attr("on_interaction_block")(py::cast(b), py::cast(a), opts_);
    }

    py::dict finalize(py::dict results) override {
        py::gil_scoped_acquire gil;
        if (py::hasattr(obj_, "finalize")) obj_.attr("finalize")(results);
        return results;
    }

    void save(py::dict results, const std::string& out_dir) override {
        py::gil_scoped_acquire gil;
        if (py::hasattr(obj_, "save_results")) {
            std::string file = out_dir + "/results.pkl";
            obj_.attr("save_results")(file, results);
        }
    }

    py::dict to_state_dict() const override {
        py::gil_scoped_acquire gil;
        if (!py::hasattr(obj_, "to_state_dict"))
            throw std::runtime_error("PythonAnalysis: missing to_state_dict()");
        return obj_.attr("to_state_dict")().cast<py::dict>();
    }

   private:
    py::object obj_;
    py::dict opts_;
};

PYBIND11_MODULE(_brass, m) {
    m.def("list_analyses", &list_analyses);
    m.def("_clear_registry", [] { AnalysisRegistry::instance().clear(); });

    py::class_<Analysis, std::shared_ptr<Analysis>>(m, "Analysis")
        .def(
            "finalize",
            [](Analysis& self, py::dict results) {
                return self.finalize(std::move(results));
            },
            py::arg("results"))
        .def(
            "save",
            [](Analysis& self, py::dict results, const std::string& out_dir) {
                self.save(std::move(results), out_dir);
            },
            py::arg("results"), py::arg("output_dir"))
        .def("to_state_dict",
             [](Analysis& self) { return self.to_state_dict(); })
        .def("set_merge_keys_dict",
             [](Analysis& self, const std::map<std::string, std::string>& kv) {
                 MergeKeySet ks;
                 for (auto& [k, v] : kv) ks.emplace_back(k, v);
                 sort_keyset(ks);
                 self.set_merge_keys(ks);
             });

    m.def(
        "create_analysis",
        [](const std::string& name) {
            return AnalysisRegistry::instance().create(name);
        },
        py::arg("name"));

    m.def(
        "register_python_analysis",
        [](const std::string& name, py::object py_factory, py::dict opts) {
            AnalysisRegistry::instance().register_factory(
                name, [name, py_factory, opts]() -> std::shared_ptr<Analysis> {
                    py::gil_scoped_acquire gil;
                    py::object obj = py_factory();
                    return std::make_shared<PythonAnalysis>(name, obj, opts);
                });
        },
        py::arg("name"), py::arg("factory"), py::arg("opts") = py::dict{});

    py::class_<ParticleBlock>(m, "ParticleBlock");
    py::class_<EndBlock>(m, "EndBlock")
        .def_readonly("event_number", &EndBlock::event_number)
        .def_readonly("impact_parameter", &EndBlock::impact_parameter);
    py::class_<InteractionBlock>(m, "InteractionBlock")
        .def_readonly("n_in", &InteractionBlock::n_in)
        .def_readonly("n_out", &InteractionBlock::n_out)
        .def_readonly("process", &InteractionBlock::process)
        .def_readonly("rho", &InteractionBlock::rho)
        .def_readonly("sigma", &InteractionBlock::sigma)
        .def_readonly("sigma_p", &InteractionBlock::sigma_p);

    // IMPORTANT: bind Accessor BEFORE DispatchingAccessor
    py::class_<Accessor, PyAccessor, std::shared_ptr<Accessor>>(m, "Accessor")
        .def(py::init<>())
        .def("on_particle_block", &Accessor::on_particle_block)
        .def("on_end_block", &Accessor::on_end_block)
        .def("on_interaction_block", &Accessor::on_interaction_block)
        .def("set_resolved_fields", &Accessor::set_resolved_fields,
             py::arg("names"),
             "Resolve and store field handles for fast repeated access")
        .def(
            "gather_block_arrays_default",
            [](const Accessor& self, const ParticleBlock& block) {
                return self.gather_arrays_default(
                    block.particles.data(), block.npart, block.particle_size);
            },
            py::arg("block"))
        .def(
            "gather_incoming_arrays_default",
            [](const Accessor& self, const InteractionBlock& ib) {
                return self.gather_arrays_default(ib.incoming.data(), ib.n_in,
                                                  ib.particle_size);
            },
            py::arg("interaction_block"))
        .def(
            "gather_outgoing_arrays_default",
            [](const Accessor& self, const InteractionBlock& ib) {
                return self.gather_arrays_default(ib.outgoing.data(), ib.n_out,
                                                  ib.particle_size);
            },
            py::arg("interaction_block"))
        .def(
            "gather_block_arrays",
            [](const Accessor& self, const ParticleBlock& block) {
                return self.gather_arrays_default(
                    block.particles.data(), block.npart, block.particle_size);
            },
            py::arg("block"), "Legacy alias for gather_block_arrays_default")
        .def(
            "gather_incoming_arrays",
            [](const Accessor& self, const InteractionBlock& ib) {
                return self.gather_arrays_default(ib.incoming.data(), ib.n_in,
                                                  ib.particle_size);
            },
            py::arg("interaction_block"),
            "Legacy alias for gather_incoming_arrays_default")
        .def(
            "gather_outgoing_arrays",
            [](const Accessor& self, const InteractionBlock& ib) {
                return self.gather_arrays_default(ib.outgoing.data(), ib.n_out,
                                                  ib.particle_size);
            },
            py::arg("interaction_block"),
            "Legacy alias for gather_outgoing_arrays_default");

    py::class_<DispatchingAccessor, Accessor,
               std::shared_ptr<DispatchingAccessor>>(m, "DispatchingAccessor")
        .def(py::init<>())
        .def("register_analysis", &DispatchingAccessor::register_analysis);

    py::class_<BinaryReader>(m, "BinaryReader")
        .def(py::init<const std::string&, const std::vector<std::string>&,
                      std::shared_ptr<Accessor>>(),
             py::arg("filename"), py::arg("quantities"), py::arg("accessor"))
        .def("read", &BinaryReader::read,
             py::call_guard<py::gil_scoped_release>());

    m.def(
        "parse_meta",
        [](const std::string& meta) {
            MergeKeySet ks = parse_merge_key(meta);
            std::map<std::string, std::string> out;
            for (auto const& mk : ks) {
                std::string v = std::visit(
                    [](auto const& x) -> std::string {
                        using T = std::decay_t<decltype(x)>;
                        if constexpr (std::is_same_v<T, std::string>) {
                            return x;
                        } else {
                            return std::to_string(x);
                        }
                    },
                    mk.value);
                out[mk.name] = v;
            }
            return out;
        },
        py::arg("meta"));
}
