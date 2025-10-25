// bindings.cpp
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
#include "binaryreader.h"  // <- match header casing

namespace py = pybind11;
// -------------------- Generic AoS helpers (DRY) --------------------
struct AoSView {
    const char* base;      // pointer to element 0
    size_t      count;     // number of elements
    size_t      stride;    // bytes between elements
};

// Gather any list of quantities into contiguous NumPy arrays (copies).
static py::list gather_aos_arrays_generic(
    const AoSView& v,
    const std::unordered_map<std::string, size_t>& layout,
    const std::vector<std::string>& names)
{
    py::list out;
    const char* base = v.base;
    const size_t n = v.count;
    const size_t stride = v.stride;

    for (const auto& name : names) {
        auto it_ty = quantity_string_map.find(name);
        if (it_ty == quantity_string_map.end())
            throw std::runtime_error("Unknown quantity: " + name);
        auto it_off = layout.find(name);
        if (it_off == layout.end())
            throw std::runtime_error("Quantity not in layout: " + name);

        const size_t off = it_off->second;

        if (it_ty->second == QuantityType::Double) {
            py::array_t<double> arr(n);
            auto a = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < n; ++i) {
                const char* p = base + i * stride + off;
                a(i) = *reinterpret_cast<const double*>(p);
            }
            out.append(py::make_tuple(name, std::move(arr)));
        } else { // Int32
            py::array_t<int32_t> arr(n);
            auto a = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < n; ++i) {
                const char* p = base + i * stride + off;
                a(i) = *reinterpret_cast<const int32_t*>(p);
            }
            out.append(py::make_tuple(name, std::move(arr)));
        }
    }
    return out;
}

// View creators for each source
static inline AoSView make_view(const ParticleBlock& b) {
    return { b.particles.data(), static_cast<size_t>(b.npart), b.particle_size };
}
static inline AoSView make_in_view(const InteractionBlock& ib) {
    return { ib.incoming.data(), ib.n_in(), ib.particle_size };
}
static inline AoSView make_out_view(const InteractionBlock& ib) {
    return { ib.outgoing.data(), ib.n_out(), ib.particle_size };
}

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

// PythonAnalysis wrapper: forwards to user Python object
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
                "PythonAnalysis requires 'merge_from(other, opts)' or '__iadd__(other)'");
        }
        return *this;
    }

    void analyze_particle_block(const ParticleBlock& b, const Accessor& a) override {
        py::gil_scoped_acquire gil;
        obj_.attr("on_particle_block")(py::cast(b), py::cast(a), opts_);
    }
    void analyze_end_block(const EndBlock& b, const Accessor& a) override {
        py::gil_scoped_acquire gil;
        obj_.attr("on_end_block")(py::cast(b), py::cast(a), opts_);
    }
    void analyze_interaction_block(const InteractionBlock& b, const Accessor& a) override {
        py::gil_scoped_acquire gil;
        // FIXED: call the proper Python hook
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
                keys[py::str(mk.name)] = std::visit([](auto const& x) { return py::cast(x); }, mk.value);
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
    // Functions
    m.def("run_analysis", &run_analysis,
          py::arg("file_and_meta"),
          py::arg("analysis_names"),
          py::arg("quantities"),
          py::arg("output_folder") = ".");

    m.def("list_analyses", &list_analyses,
          "Return the names of all registered analyses as a list of strings");

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
        py::arg("name"), py::arg("factory"), py::arg("opts") = py::dict{}
    );

    // Minimal type exposure: no fields; they’re only needed for method signatures.
    py::class_<ParticleBlock>(m, "ParticleBlock");
    py::class_<EndBlock>(m, "EndBlock")
    .def_readonly("event_number", &EndBlock::event_number)
    .def_readonly("impact_parameter", &EndBlock::impact_parameter);
    py::class_<InteractionBlock>(m, "InteractionBlock");

    // Accessor with only the APIs you want (plus callbacks)
    py::class_<Accessor, PyAccessor, std::shared_ptr<Accessor>>(m, "Accessor")
        .def(py::init<>())

        // Callback hooks (Python subclasses override these)
        .def("on_particle_block", &Accessor::on_particle_block)
        .def("on_end_block", &Accessor::on_end_block)
        .def("on_interaction_block", &Accessor::on_interaction_block)

        // --- New, minimal gather API ---
        .def("gather_block_arrays",
             [](const Accessor& self,
                const ParticleBlock& block,
                const std::vector<std::string>& names) {
                 return gather_aos_arrays_generic(
                     make_view(block), self.layout_map(), names);
             },
             py::arg("block"), py::arg("names"))

        .def("gather_incoming_arrays",
             [](const Accessor& self,
                const InteractionBlock& ib,
                const std::vector<std::string>& names) {
                 return gather_aos_arrays_generic(
                     make_in_view(ib), self.layout_map(), names);
             },
             py::arg("interaction_block"), py::arg("names"))

        .def("gather_outgoing_arrays",
             [](const Accessor& self,
                const InteractionBlock& ib,
                const std::vector<std::string>& names) {
                 return gather_aos_arrays_generic(
                     make_out_view(ib), self.layout_map(), names);
             },
             py::arg("interaction_block"), py::arg("names"));

    // BinaryReader (needed for your usage)
    py::class_<BinaryReader>(m, "BinaryReader")
        .def(py::init<const std::string&, const std::vector<std::string>&,
                      std::shared_ptr<Accessor>>())
        .def("read", &BinaryReader::read,
             py::call_guard<py::gil_scoped_release>());
}
