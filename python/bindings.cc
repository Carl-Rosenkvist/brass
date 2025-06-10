// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <filesystem>


#include "binaryreader.h"
#include "analysis.h"
#include "analysisregister.h"




namespace py = pybind11;


#include <optional>


#include <sstream>




// Helper: safe read from a span at byte offset
template <class T>
inline T read_from_span(std::span<const char> s, size_t offset) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    if (offset + sizeof(T) > s.size()) {
        throw std::out_of_range("read_from_span: offset out of range");
    }
    T val;
    std::memcpy(&val, s.data() + offset, sizeof(T));
    return val;
}

class CollectorAccessor : public Accessor {
public:
    std::unordered_map<std::string, std::vector<double>>  doubles;
    std::unordered_map<std::string, std::vector<int32_t>> ints;
    std::vector<int> event_sizes;

    void on_particle_block(const ParticleBlock& block) override {
        if (!layout) throw std::runtime_error("Layout not set");

        event_sizes.push_back(static_cast<int>(block.npart));

        for (size_t i = 0; i < block.npart; ++i) {
            std::span<const char> particle = block.particle(i);

            for (const auto& [name, info] : quantity_string_map) {
                auto it = layout->find(info.quantity);
                if (it == layout->end()) continue;

                const size_t offset = it->second;

                switch (info.type) {
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

    const std::vector<double>&  get_double_array(const std::string& name) const { return doubles.at(name); }
    const std::vector<int32_t>& get_int_array(const std::string& name)   const { return ints.at(name); }
    const std::vector<int>&     get_event_sizes()                         const { return event_sizes; }
};





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

PYBIND11_MODULE(brass, m) {

m.def("run_analysis", &run_analysis,
      py::arg("file_and_meta"),
      py::arg("analysis_name"),
      py::arg("quantities"),
      py::arg("save_output") = true,
      py::arg("print_output") = true,
      py::arg("output_folder") = ".");


    py::class_<ParticleBlock>(m, "ParticleBlock")
        .def_readonly("event_number", &ParticleBlock::event_number)
        .def_readonly("ensamble_number", &ParticleBlock::ensamble_number)
        .def_readonly("npart", &ParticleBlock::npart);

    py::class_<EndBlock>(m, "EndBlock")
        .def_readonly("event_number", &EndBlock::event_number)
        .def_readonly("impact_parameter", &EndBlock::impact_parameter);

    py::class_<Accessor, PyAccessor, std::shared_ptr<Accessor>>(m, "Accessor")
        .def(py::init<>())
        .def("on_particle_block", &Accessor::on_particle_block)
        .def("on_end_block", &Accessor::on_end_block)
        .def("get_int", &Accessor::get_int)
        .def("get_double", &Accessor::get_double);
    py::class_<BinaryReader>(m, "BinaryReader")
        .def(py::init<const std::string&, const std::vector<std::string>&, std::shared_ptr<Accessor>>())
        .def("read", &BinaryReader::read);

     py::class_<CollectorAccessor, Accessor, std::shared_ptr<CollectorAccessor>>(m, "CollectorAccessor")
        .def(py::init<>())
        .def("get_double_array", [](const CollectorAccessor& self, const std::string& name) {
            const auto& vec = self.get_double_array(name);
            return py::array(vec.size(), vec.data());
        })
        .def("get_event_sizes", [](const CollectorAccessor& self) {
            const auto& vec = self.get_event_sizes();
            return py::array(vec.size(), vec.data());
        })

        .def("get_int_array", [](const CollectorAccessor& self, const std::string& name) {
            const auto& vec = self.get_int_array(name);
            return py::array(vec.size(), vec.data());
        });

}
