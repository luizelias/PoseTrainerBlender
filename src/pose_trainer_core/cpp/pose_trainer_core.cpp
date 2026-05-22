#include "pose_trainer_core.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace pose_trainer {
namespace {

using Eigen::ComputeFullU;
using Eigen::ComputeFullV;
using Eigen::Dynamic;
using Eigen::JacobiSVD;
using Eigen::Matrix;
using Eigen::Matrix3f;
using Eigen::MatrixXf;
using Eigen::RowMajor;
using Eigen::Vector3f;
using Eigen::VectorXf;

Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(Vec3 a, float s) {
  return {a.x * s, a.y * s, a.z * s};
}

Vec3 operator/(Vec3 a, float s) {
  return {a.x / s, a.y / s, a.z / s};
}

Vec3& operator+=(Vec3& a, Vec3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length(Vec3 a) {
  return std::sqrt(dot(a, a));
}

Vector3f to_eigen(Vec3 v) {
  return {v.x, v.y, v.z};
}

Vec3 from_eigen(const Vector3f& v) {
  return {v.x(), v.y(), v.z()};
}

Vec3 mul(const Matrix3f& r, Vec3 v) {
  return from_eigen(r * to_eigen(v));
}

Vec3 centroid(const std::vector<Vec3>& points) {
  Vec3 out;
  if (points.empty()) {
    return out;
  }
  for (Vec3 p : points) {
    out += p;
  }
  return out / static_cast<float>(points.size());
}

Vec3 centroid16(const std::array<Vec3, kRepresentativeVertexCount>& points) {
  Vec3 out;
  for (Vec3 p : points) {
    out += p;
  }
  return out / static_cast<float>(kRepresentativeVertexCount);
}

float basis(float distance, float radius) {
  return 1.0f / std::sqrt(distance * distance + radius * radius);
}

std::vector<std::vector<uint32_t>> build_neighbors(
    uint32_t vertex_count,
    const std::vector<std::vector<uint32_t>>& faces) {
  std::vector<std::vector<uint32_t>> neighbors(vertex_count);
  auto add_edge = [&](uint32_t a, uint32_t b) {
    if (a >= vertex_count || b >= vertex_count) {
      throw std::invalid_argument("face index out of range");
    }
    if (a == b) {
      return;
    }
    neighbors[a].push_back(b);
    neighbors[b].push_back(a);
  };

  for (const auto& face : faces) {
    if (face.size() < 2) {
      continue;
    }
    for (size_t i = 0; i < face.size(); ++i) {
      add_edge(face[i], face[(i + 1) % face.size()]);
    }
  }

  for (auto& ring : neighbors) {
    std::sort(ring.begin(), ring.end());
    ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
  }
  return neighbors;
}

std::vector<Vec3> relax(
    const std::vector<Vec3>& points,
    const std::vector<std::vector<uint32_t>>& neighbors,
    int iterations) {
  std::vector<Vec3> current = points;
  for (int iter = 0; iter < std::max(0, iterations); ++iter) {
    std::vector<Vec3> next = current;
    for (size_t v = 0; v < current.size(); ++v) {
      const auto& ring = neighbors[v];
      if (ring.empty()) {
        continue;
      }
      Vec3 center;
      for (uint32_t n : ring) {
        center += current[n];
      }
      next[v] = center / static_cast<float>(ring.size());
    }
    current.swap(next);
  }
  return current;
}

std::vector<float> relax_weights(
    const std::vector<float>& weights,
    const std::vector<std::vector<uint32_t>>& neighbors,
    int iterations) {
  std::vector<float> current = weights;
  for (int iter = 0; iter < std::max(0, iterations); ++iter) {
    std::vector<float> next = current;
    for (size_t v = 0; v < current.size(); ++v) {
      const auto& ring = neighbors[v];
      if (ring.empty()) {
        continue;
      }
      float total = current[v];
      for (uint32_t n : ring) {
        total += current[n];
      }
      next[v] = std::clamp(total / static_cast<float>(ring.size() + 1), 0.0f, 1.0f);
    }
    current.swap(next);
  }
  return current;
}

Matrix3f procrustes_rotation(
    const std::array<Vec3, kRepresentativeVertexCount>& src,
    const std::array<Vec3, kRepresentativeVertexCount>& dst) {
  const Vec3 cs = centroid16(src);
  const Vec3 cd = centroid16(dst);

  Matrix3f h = Matrix3f::Zero();
  for (int i = 0; i < kRepresentativeVertexCount; ++i) {
    h += to_eigen(src[i] - cs) * to_eigen(dst[i] - cd).transpose();
  }
  if (h.squaredNorm() < 1.0e-20f) {
    return Matrix3f::Identity();
  }

  JacobiSVD<Matrix3f> svd(h, ComputeFullU | ComputeFullV);
  Matrix3f u = svd.matrixU();
  Matrix3f v = svd.matrixV();
  Matrix3f r = v * u.transpose();
  if (r.determinant() < 0.0f) {
    v.col(2) *= -1.0f;
    r = v * u.transpose();
  }
  return r;
}

std::array<Vec3, kRepresentativeVertexCount> gather_reps(
    const std::vector<Vec3>& points,
    const std::array<uint32_t, kRepresentativeVertexCount>& reps) {
  std::array<Vec3, kRepresentativeVertexCount> out{};
  for (int i = 0; i < kRepresentativeVertexCount; ++i) {
    out[i] = points[reps[i]];
  }
  return out;
}

std::array<Vec3, kRepresentativeVertexCount> align_points(
    const std::array<Vec3, kRepresentativeVertexCount>& src,
    const std::array<Vec3, kRepresentativeVertexCount>& dst) {
  const Vec3 cs = centroid16(src);
  const Vec3 cd = centroid16(dst);
  const Matrix3f r = procrustes_rotation(src, dst);
  std::array<Vec3, kRepresentativeVertexCount> out{};
  for (int i = 0; i < kRepresentativeVertexCount; ++i) {
    out[i] = mul(r, src[i] - cs) + cd;
  }
  return out;
}

std::vector<float> train_theta(const std::vector<Vec3>& features, int sample_count, float radius, float regularization) {
  MatrixXf phi = MatrixXf::Zero(sample_count, sample_count);
  for (int i = 0; i < sample_count; ++i) {
    for (int j = 0; j < sample_count; ++j) {
      float total = 0.0f;
      for (int r = 0; r < kRepresentativeVertexCount; ++r) {
        total += basis(length(features[i * kRepresentativeVertexCount + r] -
                              features[j * kRepresentativeVertexCount + r]), radius);
      }
      phi(i, j) = total / static_cast<float>(kRepresentativeVertexCount);
    }
  }

  MatrixXf a = phi.transpose() * phi;
  a.diagonal().array() += regularization;

  Eigen::LDLT<MatrixXf> ldlt(a);
  MatrixXf theta_matrix;
  if (ldlt.info() == Eigen::Success) {
    theta_matrix = phi * ldlt.solve(MatrixXf::Identity(sample_count, sample_count));
  } else {
    theta_matrix = phi * a.completeOrthogonalDecomposition().solve(
                             MatrixXf::Identity(sample_count, sample_count));
  }

  std::vector<float> theta_values(sample_count * sample_count, 0.0f);
  for (int i = 0; i < sample_count; ++i) {
    for (int j = 0; j < sample_count; ++j) {
      theta_values[i * sample_count + j] = theta_matrix(i, j);
    }
  }
  return theta_values;
}

std::array<uint32_t, kRepresentativeVertexCount> sample_reps(
    const std::vector<Vec3>& bind_relaxed,
    const std::vector<float>& weights) {
  std::vector<uint32_t> candidates;
  for (uint32_t i = 0; i < weights.size(); ++i) {
    if (weights[i] > 0.001f) {
      candidates.push_back(i);
    }
  }
  if (candidates.empty()) {
    candidates.push_back(0);
  }

  std::array<uint32_t, kRepresentativeVertexCount> reps{};
  reps[0] = candidates[0];
  std::vector<float> min_dist(candidates.size(), std::numeric_limits<float>::max());
  int selected_count = 1;
  for (; selected_count < kRepresentativeVertexCount && selected_count < static_cast<int>(candidates.size()); ++selected_count) {
    const uint32_t last = reps[selected_count - 1];
    for (size_t i = 0; i < candidates.size(); ++i) {
      const float d = length(bind_relaxed[candidates[i]] - bind_relaxed[last]);
      min_dist[i] = std::min(min_dist[i], d);
    }
    const auto best = std::max_element(min_dist.begin(), min_dist.end());
    reps[selected_count] = candidates[static_cast<size_t>(best - min_dist.begin())];
    *best = -1.0f;
  }
  for (int i = selected_count; i < kRepresentativeVertexCount; ++i) {
    reps[i] = reps[selected_count - 1];
  }
  return reps;
}

}  // namespace

std::vector<float> project_simplex(const std::vector<float>& weights) {
  if (weights.empty()) {
    return {};
  }
  std::vector<float> sorted = weights;
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());

  float cumulative = 0.0f;
  int rho = 0;
  for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
    cumulative += sorted[i];
    const float threshold = (cumulative - 1.0f) / static_cast<float>(i + 1);
    if (sorted[i] - threshold > 0.0f) {
      rho = i;
    }
  }

  cumulative = 0.0f;
  for (int i = 0; i <= rho; ++i) {
    cumulative += sorted[i];
  }
  const float tau = (cumulative - 1.0f) / static_cast<float>(rho + 1);

  std::vector<float> projected(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    projected[i] = std::max(weights[i] - tau, 0.0f);
  }
  return projected;
}

PoseTrainerCache train(
    const std::vector<std::vector<uint32_t>>& faces,
    const std::vector<Vec3>& bind,
    const std::vector<std::vector<Vec3>>& samples,
    const std::vector<AreaInput>& areas,
    const PoseTrainerSettings& settings) {
  if (bind.empty()) {
    throw std::invalid_argument("bind cannot be empty");
  }
  for (const auto& sample : samples) {
    if (sample.size() != bind.size()) {
      throw std::invalid_argument("sample vertex count does not match bind");
    }
  }

  PoseTrainerCache cache;
  cache.vertex_count = static_cast<uint32_t>(bind.size());
  cache.settings = settings;
  cache.faces = faces;
  cache.neighbors = build_neighbors(cache.vertex_count, faces);
  cache.bind_relaxed = relax(bind, cache.neighbors, settings.relax_iterations);

  for (const auto& sample : samples) {
    std::vector<Vec3> relaxed = relax(sample, cache.neighbors, settings.relax_iterations);
    std::vector<Vec3> delta(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) {
      delta[i] = sample[i] - relaxed[i];
    }
    cache.sample_relaxed.push_back(std::move(relaxed));
    cache.sample_deltas.push_back(std::move(delta));
  }

  const int sample_count = static_cast<int>(samples.size()) + 1;
  for (size_t area_index = 0; area_index < areas.size(); ++area_index) {
    const AreaInput& input = areas[area_index];
    if (input.weights.size() != bind.size()) {
      throw std::invalid_argument("area weights length does not match bind");
    }
    AreaModel model;
    model.name = input.name;
    model.area_id = static_cast<uint32_t>(area_index);
    model.vertex_weights = relax_weights(input.weights, cache.neighbors, settings.area_relax_iterations);
    model.reps = sample_reps(cache.bind_relaxed, model.vertex_weights);

    const auto bind_feature = gather_reps(cache.bind_relaxed, model.reps);
    model.features.resize(sample_count * kRepresentativeVertexCount);
    for (int r = 0; r < kRepresentativeVertexCount; ++r) {
      model.features[r] = bind_feature[r];
    }
    for (size_t s = 0; s < cache.sample_relaxed.size(); ++s) {
      const auto feature = gather_reps(cache.sample_relaxed[s], model.reps);
      const auto aligned = align_points(feature, bind_feature);
      for (int r = 0; r < kRepresentativeVertexCount; ++r) {
        model.features[(s + 1) * kRepresentativeVertexCount + r] = aligned[r];
      }
    }

    Vec3 min_p = bind_feature[0];
    Vec3 max_p = bind_feature[0];
    for (Vec3 p : bind_feature) {
      min_p.x = std::min(min_p.x, p.x); min_p.y = std::min(min_p.y, p.y); min_p.z = std::min(min_p.z, p.z);
      max_p.x = std::max(max_p.x, p.x); max_p.y = std::max(max_p.y, p.y); max_p.z = std::max(max_p.z, p.z);
    }
    const float diagonal = length(max_p - min_p);
    model.scale = diagonal > 1.0e-5f ? 1.0f / diagonal : 1.0f;
    for (Vec3& feature : model.features) {
      feature = feature * model.scale;
    }
    model.theta = train_theta(model.features, sample_count, settings.rbf_radius, settings.regularization);
    cache.areas.push_back(std::move(model));
  }

  return cache;
}

std::vector<Vec3> PoseTrainerCache::evaluate(
    const std::vector<Vec3>& animated,
    const std::vector<float>& vertex_mask,
    float envelope,
    int solve_iterations_override) const {
  if (animated.size() != vertex_count) {
    throw std::invalid_argument("animated vertex count does not match cache");
  }
  if (!vertex_mask.empty() && vertex_mask.size() != vertex_count) {
    throw std::invalid_argument("vertex mask length does not match cache");
  }

  std::vector<Vec3> current = animated;
  const int iterations = std::max(1, solve_iterations_override > 0 ? solve_iterations_override : settings.solve_iterations);
  const int sample_count = static_cast<int>(sample_deltas.size()) + 1;

  for (int iter = 0; iter < iterations; ++iter) {
    const std::vector<Vec3> relaxed = relax(current, neighbors, settings.relax_iterations);
    std::vector<Vec3> pose_delta(vertex_count);
    for (uint32_t v = 0; v < vertex_count; ++v) {
      pose_delta[v] = animated[v] - relaxed[v];
    }

    std::vector<Vec3> accum(vertex_count);
    std::vector<float> total_area_weight(vertex_count, 0.0f);

    for (const AreaModel& area : areas) {
      const auto current_feature = gather_reps(relaxed, area.reps);
      const auto bind_feature = gather_reps(bind_relaxed, area.reps);
      auto aligned_current = align_points(current_feature, bind_feature);
      for (Vec3& p : aligned_current) {
        p = p * area.scale;
      }

      std::vector<float> phi(sample_count, 0.0f);
      for (int s = 0; s < sample_count; ++s) {
        float total = 0.0f;
        for (int r = 0; r < kRepresentativeVertexCount; ++r) {
          total += basis(length(aligned_current[r] - area.features[s * kRepresentativeVertexCount + r]), settings.rbf_radius);
        }
        phi[s] = total / static_cast<float>(kRepresentativeVertexCount);
      }

      Eigen::Map<const Matrix<float, Dynamic, Dynamic, RowMajor>> theta_matrix(
          area.theta.data(), sample_count, sample_count);
      Eigen::Map<const VectorXf> phi_vector(phi.data(), sample_count);
      const VectorXf raw_vector = theta_matrix * phi_vector;
      std::vector<float> raw(sample_count, 0.0f);
      for (int i = 0; i < sample_count; ++i) {
        raw[i] = raw_vector(i);
      }
      const std::vector<float> activation = project_simplex(raw);

      std::vector<Matrix3f> rotations(sample_deltas.size());
      for (size_t s = 0; s < sample_deltas.size(); ++s) {
        const auto sample_feature = gather_reps(sample_relaxed[s], area.reps);
        rotations[s] = procrustes_rotation(sample_feature, current_feature);
      }

      for (uint32_t v = 0; v < vertex_count; ++v) {
        const float area_w = area.vertex_weights[v];
        if (area_w <= 0.0f) {
          continue;
        }
        total_area_weight[v] += area_w;
        accum[v] += pose_delta[v] * (area_w * activation[0]);
        for (size_t s = 0; s < sample_deltas.size(); ++s) {
          accum[v] += mul(rotations[s], sample_deltas[s][v]) * (area_w * activation[s + 1]);
        }
      }
    }

    std::vector<Vec3> output(vertex_count);
    for (uint32_t v = 0; v < vertex_count; ++v) {
      const float uncovered = std::max(1.0f - total_area_weight[v], 0.0f);
      accum[v] += pose_delta[v] * uncovered;
      const Vec3 corrected = relaxed[v] + accum[v];
      const float mask = vertex_mask.empty() ? 1.0f : vertex_mask[v];
      const float blend = envelope * mask;
      output[v] = animated[v] * (1.0f - blend) + corrected * blend;
    }
    current.swap(output);
  }

  return current;
}

}  // namespace pose_trainer
