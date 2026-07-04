#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "binaryreader.h"
#include "histogram_runner.h"

namespace py = pybind11;

namespace {

py::object block_to_python(brass::Block&& block) {
    return std::visit(
        [](auto&& value) -> py::object { return py::cast(std::move(value)); },
        std::move(block));
}

py::array particle_column_to_numpy(const brass::Particles& particles,
                                   const std::string& name) {
    const auto& q = particles.quantity(name);

    if (q.size == sizeof(double)) {
        const auto& values = particles.column<double>(name);

        return py::array_t<double>(values.size(), values.data(),
                                   py::cast(&particles));
    }

    if (q.size == sizeof(int32_t)) {
        const auto& values = particles.column<int32_t>(name);

        return py::array_t<int32_t>(values.size(), values.data(),
                                    py::cast(&particles));
    }

    throw std::runtime_error("unsupported quantity size: " + name);
}

py::array_t<double> histogram_values_to_numpy(
    const brass::HistogramResult& result) {
    std::vector<py::ssize_t> shape;
    shape.reserve(result.shape.size());

    for (const auto size : result.shape) {
        shape.push_back(static_cast<py::ssize_t>(size));
    }

    std::vector<py::ssize_t> strides(shape.size());

    if (!shape.empty()) {
        strides[0] = static_cast<py::ssize_t>(sizeof(double));

        for (std::size_t i = 1; i < shape.size(); ++i) {
            strides[i] = strides[i - 1] * shape[i - 1];
        }
    }

    return py::array_t<double>(shape, strides, result.values.data(),
                               py::cast(&result));
}

}  // namespace

PYBIND11_MODULE(_brass, m) {
    m.doc() = "BRASS binary reader bindings";

    py::class_<brass::Header>(m, "Header")
        .def_readonly("format_version", &brass::Header::format_version)
        .def_readonly("format_variant", &brass::Header::format_variant)
        .def_readonly("smash_version", &brass::Header::smash_version);

    py::class_<brass::Particles>(m, "Particles")
        .def("size", &brass::Particles::size)
        .def("particle_size", &brass::Particles::particle_size)
        .def("empty", &brass::Particles::empty)
        .def("column",
             [](const brass::Particles& particles, const std::string& name) {
                 return particle_column_to_numpy(particles, name);
             })
        .def("columns", [](const brass::Particles& particles) {
            py::dict out;

            for (const auto& q : particles.layout().offsets) {
                out[py::str(q.name)] =
                    particle_column_to_numpy(particles, q.name);
            }

            return out;
        });

    py::class_<brass::ParticleBlock>(m, "ParticleBlock")
        .def_readonly("event_number", &brass::ParticleBlock::event_number)
        .def_readonly("ensemble_number", &brass::ParticleBlock::ensemble_number)
        .def_readonly("particles", &brass::ParticleBlock::particles,
                      py::return_value_policy::reference_internal);

    py::class_<brass::EndBlock>(m, "EndBlock")
        .def_readonly("event_number", &brass::EndBlock::event_number)
        .def_readonly("ensemble_number", &brass::EndBlock::ensemble_number)
        .def_readonly("impact_parameter", &brass::EndBlock::impact_parameter)
        .def_readonly("empty", &brass::EndBlock::empty);

    py::class_<brass::InteractionBlock>(m, "InteractionBlock")
        .def_readonly("n_in", &brass::InteractionBlock::n_in)
        .def_readonly("n_out", &brass::InteractionBlock::n_out)
        .def_readonly("rho", &brass::InteractionBlock::rho)
        .def_readonly("sigma", &brass::InteractionBlock::sigma)
        .def_readonly("sigma_p", &brass::InteractionBlock::sigma_p)
        .def_readonly("process", &brass::InteractionBlock::process)
        .def_readonly("incoming", &brass::InteractionBlock::incoming,
                      py::return_value_policy::reference_internal)
        .def_readonly("outgoing", &brass::InteractionBlock::outgoing,
                      py::return_value_policy::reference_internal);

    m.def("particle_size_from_quantities",
          &brass::particle_size_from_quantities, py::arg("quantities"));

    py::class_<brass::BinaryReader>(m, "BinaryReader")
        .def(py::init<const std::string&, std::vector<std::string>>(),
             py::arg("filename"), py::arg("quantities"))
        .def_property_readonly(
            "header",
            [](const brass::BinaryReader& reader) -> const brass::Header& {
                return reader.header();
            },
            py::return_value_policy::reference_internal)
        .def_property_readonly("particle_blocks_read",
                               &brass::BinaryReader::particle_blocks_read)
        .def_property_readonly("end_blocks_read",
                               &brass::BinaryReader::end_blocks_read)
        .def_property_readonly("interaction_blocks_read",
                               &brass::BinaryReader::interaction_blocks_read)

        .def("read", [](brass::BinaryReader& reader) -> py::object {
            auto block = reader.read();

            if (!block) {
                return py::none();
            }

            return block_to_python(std::move(*block));
        });

    py::class_<brass::RegularAxis>(m, "RegularAxis")
        .def(py::init<int, double, double>(), py::arg("bins"), py::arg("lower"),
             py::arg("upper"))
        .def_readwrite("bins", &brass::RegularAxis::bins)
        .def_readwrite("lower", &brass::RegularAxis::lower)
        .def_readwrite("upper", &brass::RegularAxis::upper);

    py::class_<brass::VariableAxis>(m, "VariableAxis")
        .def(py::init<std::vector<double>>(), py::arg("edges"))
        .def_readwrite("edges", &brass::VariableAxis::edges);

    py::class_<brass::IntegerAxis>(m, "IntegerAxis")
        .def(py::init<int, int>(), py::arg("lower"), py::arg("upper"))
        .def_readwrite("lower", &brass::IntegerAxis::lower)
        .def_readwrite("upper", &brass::IntegerAxis::upper);

    py::class_<brass::HistogramResult>(m, "HistogramResult")
        .def_property_readonly("values",
                               [](const brass::HistogramResult& result) {
                                   return histogram_values_to_numpy(result);
                               })
        .def_readonly("edges", &brass::HistogramResult::edges)
        .def_readonly("shape", &brass::HistogramResult::shape);

    m.def("histogram", &brass::histogram_reader, py::arg("reader"),
          py::arg("histogram_quantities"), py::arg("axes"));

    m.def("histogram", &brass::histogram_particles, py::arg("particles"),
          py::arg("histogram_quantities"), py::arg("axes"));

    m.def("histograms_by", &brass::histograms_by_reader, py::arg("reader"),
          py::arg("histogram_quantities"), py::arg("axes"), py::arg("by"),
          py::arg("group_values"));

    m.def("histograms_by", &brass::histograms_by_particles,
          py::arg("particles"), py::arg("histogram_quantities"),
          py::arg("axes"), py::arg("by"), py::arg("group_values"));
}
