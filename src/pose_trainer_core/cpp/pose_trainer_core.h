#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pose_trainer {

constexpr int kRepresentativeVertexCount = 16;

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct PoseTrainerSettings {
  int relax_iterations = 10;
  int solve_iterations = 1;
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
  std::vector<Vec3> features;  // sample-major: sample * REP + rep
  float scale = 1.0f;
  std::vector<float> theta;  // sample_count x sample_count
};

struct PoseTrainerCache {
  uint32_t vertex_count = 0;
  PoseTrainerSettings settings;
  std::vector<std::vector<uint32_t>> faces;
  std::vector<std::vector<uint32_t>> neighbors;
  std::vector<Vec3> bind_relaxed;
  std::vector<std::vector<Vec3>> sample_relaxed;
  std::vector<std::vector<Vec3>> sample_deltas;
  std::vector<AreaModel> areas;

  std::vector<Vec3> evaluate(
      const std::vector<Vec3>& animated,
      const std::vector<float>& vertex_mask,
      float envelope,
      int solve_iterations_override = 0) const;
};

PoseTrainerCache train(
    const std::vector<std::vector<uint32_t>>& faces,
    const std::vector<Vec3>& bind,
    const std::vector<std::vector<Vec3>>& samples,
    const std::vector<AreaInput>& areas,
    const PoseTrainerSettings& settings);

std::vector<float> project_simplex(const std::vector<float>& weights);

}  // namespace pose_trainer
