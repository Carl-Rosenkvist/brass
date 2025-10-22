// bindings.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>

#include "analysis.h"
#include "analysisregister.h"
#include "binaryreader.h"  // <- match header casing

namespace py = pybind11;

#include <cstring>  // for std::memcpy
#include <optional>
#include <span>  // for std::span
#include <sstream>
#include <type_traits>  // for std::is_trivially_copyable_v

// Helper: safe read from a span at byte offset
template <class T>
inline T read_from_span(std::span<const char> s, size_t offset) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");
    if (offset + sizeof(T) > s.size()) {
        throw std::out_of_range("read_from_span: offset out of range");
    }
    T val;
    std::memcpy(&val, s.data() + offset, sizeof(T));
    return val;
}

class CollectorAccessor : public Accessor {
   public:
    std::unordered_map<std::string, std::vector<double>> doubles;
    std::unordered_map<std::string, std::vector<int32_t>> ints;
    std::vector<int> event_sizes;

    void on_particle_block(const ParticleBlock& block) override {
        if (!layout) throw std::runtime_error("Layout not set");

        event_sizes.push_back(static_cast<int>(block.npart));

        for (size_t i = 0; i < block.npart; ++i) {
            std::span<const char> particle = block.particle(i);

            // quantity_string_map is now: name -> QuantityType
            for (const auto& [name, qtype] : quantity_string_map) {
                auto it = layout->find(name);  // layout is name -> offset
                if (it == layout->end()) continue;

                const size_t offset = it->second;

                switch (qtype) {
                    case QuantityType::Double: {
                        double v = read_from_span<double>(particle, offset);
                        doubles[name].push_back(v);
                        break;
                    }
                    case QuantityType::Int32: {
                        int32_t v = read_from_span<int32_t>(particle, offset);
                        ints[name].push_back(v);
                        break;
                    }
                    default:
                        throw std::logic_error("Unknown QuantityType");
                }
            }
        }
    }

    const std::vector<double>& get_double_array(const std::string& name) const {
        return doubles.at(name);
    }
    const std::vector<int32_t>& get_int_array(const std::string& name) const {
        return ints.at(name);
    }
    const std::vector<int>& get_event_sizes() const { return event_sizes; }
};

std::vector<std::string> list_analyses() {
    return AnalysisRegistry::instance().list_registered();
}

// Trampoline to call Python overrides
class PyAccessor : public Accessor {
   public:
    using Accessor::Accessor;

    void on_particle_block(const ParticleBlock& block) override {
        PYBIND11_OVERRIDE(void, Accessor, on_particle_block, block);
    }

    void on_end_block(const EndBlock& block) override {
        PYBIND11_OVERRIDE(void, Accessor, on_end_block, block);
    }
};

static py::array make_block_array(
    const ParticleBlock& block,
    const std::unordered_map<std::string, size_t>& layout,
    const std::string& name, QuantityType qtype) {
    auto it = layout.find(name);
    if (it == layout.end())
        throw std::runtime_error("Quantity not in layout: " + name);

    const size_t offset = it->second;
    const py::ssize_t N = static_cast<py::ssize_t>(block.npart);
    const py::ssize_t stride = static_cast<py::ssize_t>(block.particle_size);

    // Empty case: return zero-length array of the right dtype
    if (N == 0) {
        if (qtype == QuantityType::Double)
            return py::array_t<double>(0);
        else
            return py::array_t<int32_t>(0);
    }

    const char* base_bytes = block.particle(0).data() + offset;

    if (qtype == QuantityType::Double) {
        auto* data = reinterpret_cast<const double*>(base_bytes);
        // dtype, shape, strides, data ptr, base (keep block alive)
        return py::array(py::dtype::of<double>(), {N}, {stride},
                         const_cast<double*>(data), py::cast(block));
    } else if (qtype == QuantityType::Int32) {
        auto* data = reinterpret_cast<const int32_t*>(base_bytes);
        return py::array(py::dtype::of<int32_t>(), {N}, {stride},
                         const_cast<int32_t*>(data), py::cast(block));
    }
    throw std::logic_error("Unsupported QuantityType");
}

// Helper: gather any list of quantities into contiguous arrays for one block.
static py::list gather_block_arrays_generic(
    const ParticleBlock& b,
    const std::unordered_map<std::string, size_t>& layout,
    const std::vector<std::string>& names) {
    const size_t n = b.npart, stride = b.particle_size;
    const char* base = b.particles.data();

    py::list out;  // no reserve() on Python lists

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
        } else {  // Int32
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

// helpers (e.g., near PythonAnalysis)
static pybind11::object mkvalue_to_py(const MergeKeyValue& v) {
    return std::visit([](auto const& x) { return pybind11::cast(x); }, v);
}

static pybind11::dict mergekeys_to_pydict(const MergeKeySet& ks) {
    pybind11::dict d;
    for (auto const& mk : ks) {
        d[pybind11::str(mk.name)] = mkvalue_to_py(mk.value);
    }
    return d;
}
class PythonAnalysis : public Analysis {
   public:
    PythonAnalysis(const std::string& name, pybind11::object py_obj,
                   pybind11::dict opts)
        : Analysis(name), obj_(std::move(py_obj)), opts_(std::move(opts)) {}

    // Merge two PythonAnalysis objects
    Analysis& operator+=(const Analysis& other) override {
        auto* o = dynamic_cast<const PythonAnalysis*>(&other);
        if (!o) throw std::runtime_error("PythonAnalysis: merge mismatch");
        pybind11::gil_scoped_acquire gil;
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
        pybind11::gil_scoped_acquire gil;
        obj_.attr("on_particle_block")(py::cast(b), py::cast(a), opts_);
    }

    void finalize() override {
        pybind11::gil_scoped_acquire gil;
        if (py::hasattr(obj_, "finalize")) obj_.attr("finalize")(opts_);
    }

    void save(const std::string& out_dir) override {
        pybind11::gil_scoped_acquire gil;
        if (py::hasattr(obj_, "save")) {
            obj_.attr("save")(out_dir, mergekeys_to_pydict(this->keys), opts_);
        }
    }

   private:
    py::object obj_;
    py::dict opts_;
};

PYBIND11_MODULE(_brass, m) {
    // --- Functions ---
    m.def("run_analysis", &run_analysis, py::arg("file_and_meta"),
          py::arg("analysis_names"), py::arg("quantities"),
          py::arg("output_folder") = ".");

    m.def("list_analyses", &list_analyses,
          "Return the names of all registered analyses as a list of strings");

    m.def("_clear_registry", [] { AnalysisRegistry::instance().clear(); });
    // --- ParticleBlock / EndBlock ---
    py::class_<ParticleBlock>(m, "ParticleBlock")
        .def_readonly("event_number", &ParticleBlock::event_number)
        .def_readonly("ensamble_number", &ParticleBlock::ensamble_number)
        .def_readonly("npart", &ParticleBlock::npart)
        .def_readonly("particle_size", &ParticleBlock::particle_size);

    py::class_<EndBlock>(m, "EndBlock")
        .def_readonly("event_number", &EndBlock::event_number)
        .def_readonly("impact_parameter", &EndBlock::impact_parameter);

    m.def(
        "register_python_analysis",
        [](const std::string& name, py::object py_factory, py::dict opts) {
            AnalysisRegistry::instance().register_factory(
                name, [name, py_factory, opts]() -> std::shared_ptr<Analysis> {
                    pybind11::gil_scoped_acquire gil;
                    py::object obj = py_factory();
                    return std::make_shared<PythonAnalysis>(name, obj, opts);
                });
        },
        py::arg("name"), py::arg("factory"), py::arg("opts") = py::dict{}
        // default empty dict, also enables keyword
    );

    py::class_<Accessor, PyAccessor, std::shared_ptr<Accessor>>(m, "Accessor")
        .def(py::init<>())
        .def("on_particle_block", &Accessor::on_particle_block)
        .def("on_end_block", &Accessor::on_end_block)
        .def("get_int", &Accessor::get_int)
        .def("get_double", &Accessor::get_double)
        // zero-copy strided views over AoS
        .def("get_block_double_array",
             [](const Accessor& self, const ParticleBlock& block,
                const std::string& name) {
                 auto itq = quantity_string_map.find(name);
                 if (itq == quantity_string_map.end())
                     throw std::runtime_error("Unknown quantity: " + name);
                 if (itq->second != QuantityType::Double)
                     throw std::runtime_error("Not a double quantity: " + name);
                 return make_block_array(block, self.layout_map(), name,
                                         QuantityType::Double);
             })
        .def("get_block_int_array",
             [](const Accessor& self, const ParticleBlock& block,
                const std::string& name) {
                 auto itq = quantity_string_map.find(name);
                 if (itq == quantity_string_map.end())
                     throw std::runtime_error("Unknown quantity: " + name);
                 if (itq->second != QuantityType::Int32)
                     throw std::runtime_error("Not an int32 quantity: " + name);
                 return make_block_array(block, self.layout_map(), name,
                                         QuantityType::Int32);
             })
        // generic contiguous gather (no hardcoding, faster NumPy)
        .def("gather_block_arrays", [](const Accessor& self,
                                       const ParticleBlock& block,
                                       const std::vector<std::string>& names) {
            return gather_block_arrays_generic(block, self.layout_map(), names);
        });  // <-- end of class_ chain (note single semicolon)

    // --- BinaryReader (release the GIL during heavy read) ---
    py::class_<BinaryReader>(m, "BinaryReader")
        .def(py::init<const std::string&, const std::vector<std::string>&,
                      std::shared_ptr<Accessor>>())
        .def("read", &BinaryReader::read,
             py::call_guard<py::gil_scoped_release>());

    // --- CollectorAccessor: return NumPy views with a base to keep memory
    // alive ---
    py::class_<CollectorAccessor, Accessor, std::shared_ptr<CollectorAccessor>>(
        m, "CollectorAccessor")
        .def(py::init<>())
        .def("get_double_array",
             [](const CollectorAccessor& self, const std::string& name) {
                 const auto& v = self.get_double_array(name);
                 return py::array(py::dtype::of<double>(),
                                  {static_cast<py::ssize_t>(v.size())},
                                  {static_cast<py::ssize_t>(sizeof(double))},
                                  const_cast<double*>(v.data()),
                                  py::cast(self));
             })
        .def("get_int_array",
             [](const CollectorAccessor& self, const std::string& name) {
                 const auto& v = self.get_int_array(name);
                 return py::array(py::dtype::of<int32_t>(),
                                  {static_cast<py::ssize_t>(v.size())},
                                  {static_cast<py::ssize_t>(sizeof(int32_t))},
                                  const_cast<int32_t*>(v.data()),
                                  py::cast(self));
             })
        .def("get_event_sizes", [](const CollectorAccessor& self) {
            const auto& v = self.get_event_sizes();
            return py::array(py::dtype::of<int>(),
                             {static_cast<py::ssize_t>(v.size())},
                             {static_cast<py::ssize_t>(sizeof(int))},
                             const_cast<int*>(v.data()), py::cast(self));
        });
}
