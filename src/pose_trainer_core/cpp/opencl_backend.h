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

bool opencl_runtime_available();
std::string opencl_runtime_status();

}  // namespace pose_trainer
