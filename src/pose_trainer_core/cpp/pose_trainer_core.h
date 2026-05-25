#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pose_trainer {

struct OpenClRuntime;

constexpr int kRepresentativeVertexCount = 16;

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct PoseTrainerSettings {
  int relax_iterations = 10;
  int area_relax_iterations = 0;
  int solve_iterations = 1;
  int runtime_backend = 0;  // 0=auto, 1=CPU, 2=OpenCL
  float rbf_radius = 0.1f;
  float regularization = 0.001f;
};

struct AreaInput {
  std::string name;
  std::vector<float> weights;
};

struct AreaModel {
  std::string name;
  uint32_t area_id = 0;
  std::array<uint32_t, kRepresentativeVertexCount> reps{};
  std::vector<float> vertex_weights;
  std::vector<uint32_t> active_vertices;
  std::vector<float> active_weights;
  std::vector<Vec3> features;  // sample-major: sample * REP + rep
  float scale = 1.0f;
  std::vector<float> theta;  // sample_count x sample_count
};

struct PoseTrainerCache {
  uint32_t vertex_count = 0;
  PoseTrainerSettings settings;
  std::vector<std::vector<uint32_t>> faces;
  std::vector<std::vector<uint32_t>> neighbors;
  std::vector<uint32_t> halfedge_twin;
  std::vector<uint32_t> halfedge_next;
  std::vector<uint32_t> halfedge_vertex;
  std::vector<uint32_t> vertex_halfedge;
  std::vector<Vec3> bind_relaxed;
  std::vector<std::vector<Vec3>> sample_relaxed;
  std::vector<std::vector<Vec3>> sample_deltas;
  std::vector<AreaModel> areas;
  std::vector<float> total_area_weights;
  std::vector<uint32_t> neighbor_offsets;
  std::vector<uint32_t> neighbor_indices;
  std::vector<uint32_t> membership_offsets;
  std::vector<uint32_t> membership_area_ids;
  std::vector<float> membership_weights;
  std::vector<uint32_t> tangent_frame_offsets;
  std::vector<uint32_t> tangent_frame_center_indices;
  std::vector<uint32_t> tangent_frame_neighbor_a;
  std::vector<uint32_t> tangent_frame_neighbor_b;
  std::vector<float> sample_base_frame_inverse_flat;
  std::vector<uint32_t> sample_base_frame_valid_flat;
  std::vector<float> sample_deltas_flat;
  std::vector<uint32_t> rep_indices_flat;
  std::vector<float> feature_values_flat;
  std::vector<float> theta_values_flat;
  std::vector<float> scale_values;
  uint64_t opencl_static_key = 0;
  mutable std::shared_ptr<OpenClRuntime> opencl_runtime;
  mutable std::string last_backend = "CPU";
  mutable std::string last_opencl_timing;

  std::vector<Vec3> evaluate(
      const std::vector<Vec3>& animated,
      const std::vector<float>& vertex_mask,
      float envelope,
      int solve_iterations_override = 0,
      int runtime_backend_override = -1,
      bool profile_timing = false) const;
};

PoseTrainerCache train(
    const std::vector<std::vector<uint32_t>>& faces,
    const std::vector<Vec3>& bind,
    const std::vector<std::vector<Vec3>>& samples,
    const std::vector<AreaInput>& areas,
    const PoseTrainerSettings& settings);

std::vector<float> project_simplex(const std::vector<float>& weights);
bool opencl_is_available();
std::string opencl_status();

}  // namespace pose_trainer
