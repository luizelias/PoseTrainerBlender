#include "pose_trainer_core.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace pose_trainer;

namespace {

std::vector<Vec3> array_to_vec3(py::array_t<float, py::array::c_style | py::array::forcecast> array, const char* name) {
  py::buffer_info info = array.request();
  if (info.ndim != 2 || info.shape[1] != 3) {
    throw std::invalid_argument(std::string(name) + " must be an Nx3 float array");
  }
  const auto* data = static_cast<const float*>(info.ptr);
  std::vector<Vec3> out(static_cast<size_t>(info.shape[0]));
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
  }
  return out;
}

py::array_t<float> vec3_to_array(const std::vector<Vec3>& values) {
  py::array_t<float> out({static_cast<py::ssize_t>(values.size()), static_cast<py::ssize_t>(3)});
  py::buffer_info info = out.request();
  auto* data = static_cast<float*>(info.ptr);
  for (size_t i = 0; i < values.size(); ++i) {
    data[i * 3] = values[i].x;
    data[i * 3 + 1] = values[i].y;
    data[i * 3 + 2] = values[i].z;
  }
  return out;
}

std::vector<std::vector<uint32_t>> parse_faces(py::object faces) {
  std::vector<std::vector<uint32_t>> out;
  for (py::handle face_handle : faces) {
    std::vector<uint32_t> face;
    for (py::handle index_handle : face_handle) {
      face.push_back(index_handle.cast<uint32_t>());
    }
    out.push_back(std::move(face));
  }
  return out;
}

std::vector<std::vector<Vec3>> parse_samples(py::object samples) {
  std::vector<std::vector<Vec3>> out;
  int index = 0;
  for (py::handle sample_handle : samples) {
    out.push_back(array_to_vec3(py::cast<py::array_t<float>>(sample_handle), ("sample " + std::to_string(index)).c_str()));
    index++;
  }
  return out;
}

std::vector<AreaInput> parse_areas(py::object areas) {
  std::vector<AreaInput> out;
  int index = 0;
  for (py::handle area_handle : areas) {
    py::dict area = py::cast<py::dict>(area_handle);
    AreaInput input;
    if (area.contains("name")) {
      input.name = py::cast<std::string>(area["name"]);
    } else {
      input.name = "Area " + std::to_string(index + 1);
    }
    py::array_t<float, py::array::c_style | py::array::forcecast> weights = py::cast<py::array_t<float>>(area["weights"]);
    py::buffer_info info = weights.request();
    if (info.ndim != 1) {
      throw std::invalid_argument("area weights must be a 1D float array");
    }
    const auto* data = static_cast<const float*>(info.ptr);
    input.weights.assign(data, data + info.shape[0]);
    out.push_back(std::move(input));
    index++;
  }
  return out;
}

std::vector<float> parse_mask(py::object mask) {
  if (mask.is_none()) {
    return {};
  }
  py::array_t<float, py::array::c_style | py::array::forcecast> array = py::cast<py::array_t<float>>(mask);
  py::buffer_info info = array.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("vertex_mask must be a 1D float array");
  }
  const auto* data = static_cast<const float*>(info.ptr);
  return std::vector<float>(data, data + info.shape[0]);
}

PoseTrainerCache train_py(
    py::object faces,
    py::array_t<float, py::array::c_style | py::array::forcecast> bind,
    py::object samples,
    py::object areas,
    const PoseTrainerSettings& settings) {
  return train(
      parse_faces(faces),
      array_to_vec3(bind, "bind"),
      parse_samples(samples),
      parse_areas(areas),
      settings);
}

py::array_t<float> evaluate_py(
    const PoseTrainerCache& cache,
    py::array_t<float, py::array::c_style | py::array::forcecast> animated,
    py::object vertex_mask,
    float envelope,
    int solve_iterations,
    int runtime_backend,
    bool profile_timing) {
  return vec3_to_array(cache.evaluate(
      array_to_vec3(animated, "animated"),
      parse_mask(vertex_mask),
      envelope,
      solve_iterations,
      runtime_backend,
      profile_timing));
}

}  // namespace

PYBIND11_MODULE(_pose_trainer_core, m) {
  m.doc() = "Array-based Pose Trainer C++ core";

  py::class_<PoseTrainerSettings>(m, "PoseTrainerSettings")
      .def(py::init<>())
      .def_readwrite("relax_iterations", &PoseTrainerSettings::relax_iterations)
      .def_readwrite("area_relax_iterations", &PoseTrainerSettings::area_relax_iterations)
      .def_readwrite("solve_iterations", &PoseTrainerSettings::solve_iterations)
      .def_readwrite("runtime_backend", &PoseTrainerSettings::runtime_backend)
      .def_readwrite("rbf_radius", &PoseTrainerSettings::rbf_radius)
      .def_readwrite("regularization", &PoseTrainerSettings::regularization);

  py::class_<PoseTrainerCache>(m, "PoseTrainerCache")
      .def_property_readonly("vertex_count", [](const PoseTrainerCache& cache) { return cache.vertex_count; })
      .def_property_readonly("last_backend", [](const PoseTrainerCache& cache) { return cache.last_backend; })
      .def_property_readonly("last_opencl_timing", [](const PoseTrainerCache& cache) { return cache.last_opencl_timing; })
      .def("evaluate", &evaluate_py,
           py::arg("animated"),
           py::arg("vertex_mask") = py::none(),
           py::arg("envelope") = 1.0f,
           py::arg("solve_iterations") = 0,
           py::arg("runtime_backend") = -1,
           py::arg("profile_timing") = false);

  m.def("train", &train_py,
        py::arg("faces"),
        py::arg("bind"),
        py::arg("samples"),
        py::arg("areas"),
        py::arg("settings") = PoseTrainerSettings{});

  m.def("project_simplex", &project_simplex, py::arg("weights"));
  m.def("opencl_available", &opencl_is_available);
  m.def("opencl_status", &opencl_status);
}
