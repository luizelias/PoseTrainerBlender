#pragma once

#include "pose_trainer_core.h"

namespace pose_trainer {

std::vector<Vec3> evaluate_opencl(
    const PoseTrainerCache& cache,
    const std::vector<Vec3>& animated,
    const std::vector<float>& vertex_mask,
    float envelope,
    int solve_iterations,
    bool profile_timing);

std::vector<Vec3> relax_opencl(
    const PoseTrainerCache& cache,
    const std::vector<Vec3>& points,
    int iterations);

void train_area_models_opencl(
    PoseTrainerCache& cache,
    const std::vector<float>& rep_positions,
    int sample_count);

bool opencl_runtime_available();
std::string opencl_runtime_status();

}  // namespace pose_trainer
