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

// PythonAnalysis wrapper
class PythonAnalysis : public Analysis {
   public:
    PythonAnalysis(const std::string& name, py::object py_obj, py::dict opts)
        : Analysis(name), obj_(std::move(py_obj)), opts_(std::move(opts)) {}

    Analysis& operator+=(const Analysis& other) override {
        auto* o = dynamic_cast<const PythonAnalysis*>(&other);
        if (!o) throw std::runtime_error("PythonAnalysis: merge mismatch");
        py::gil_scoped_acquire gil;
        if (py::hasattr(obj_, "merge_from")) {
            obj_.attr("merge_from")(o->obj_, opts_);
        } else if (py::hasattr(obj_, "__iadd__")) {
            py::object r = obj_.attr("__iadd__")(o->obj_);
            if (!r.is_none()) obj_ = std::move(r);
        } else {
            throw std::runtime_error(
                "PythonAnalysis requires 'merge_from(other, opts)' or "
                "'__iadd__(other)'");
        }
        return *this;
    }

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

    void finalize() override {
        py::gil_scoped_acquire gil;
        if (py::hasattr(obj_, "finalize")) obj_.attr("finalize")(opts_);
    }

    void save(const std::string& out_dir) override {
        py::gil_scoped_acquire gil;
        if (py::hasattr(obj_, "save")) {
            py::dict keys;
            for (auto const& mk : this->keys) {
                keys[py::str(mk.name)] = std::visit(
                    [](auto const& x) { return py::cast(x); }, mk.value);
            }
            obj_.attr("save")(out_dir, keys, opts_);
        }
    }

   private:
    py::object obj_;
    py::dict opts_;
};

// -------------------- Module --------------------
PYBIND11_MODULE(_brass, m) {
    // Analysis API
    m.def("run_analysis", &run_analysis, py::arg("file_and_meta"),
          py::arg("analysis_names"), py::arg("quantities"),
          py::arg("output_folder") = ".");

    m.def("list_analyses", &list_analyses);
    m.def("_clear_registry", [] { AnalysisRegistry::instance().clear(); });

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

    // Minimal type exposure
    py::class_<ParticleBlock>(m, "ParticleBlock");
    py::class_<EndBlock>(m, "EndBlock")
        .def_readonly("event_number", &EndBlock::event_number)
        .def_readonly("impact_parameter", &EndBlock::impact_parameter);
    py::class_<InteractionBlock>(m, "InteractionBlock")
        .def_readonly("process", &InteractionBlock::process);

    // Accessor
    py::class_<Accessor, PyAccessor, std::shared_ptr<Accessor>>(m, "Accessor")
        .def(py::init<>())

        // Hooks for Python subclassing
        .def("on_particle_block", &Accessor::on_particle_block)
        .def("on_end_block", &Accessor::on_end_block)
        .def("on_interaction_block", &Accessor::on_interaction_block)

        // Let Python override or manually set resolved fields
        .def("set_resolved_fields", &Accessor::set_resolved_fields,
             py::arg("names"),
             "Resolve and store field handles for fast repeated access")

        // New fast default gather API
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

    // BinaryReader
    py::class_<BinaryReader>(m, "BinaryReader")
        .def(py::init<const std::string&, const std::vector<std::string>&,
                      std::shared_ptr<Accessor>>(),
             py::arg("filename"), py::arg("quantities"), py::arg("accessor"))
        .def("read", &BinaryReader::read,
             py::call_guard<py::gil_scoped_release>());
}
