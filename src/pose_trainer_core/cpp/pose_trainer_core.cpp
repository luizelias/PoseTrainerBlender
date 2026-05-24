#include "pose_trainer_core.h"
#include "opencl_backend.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

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

constexpr uint32_t kBoundaryHalfedge = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kMaxFaceVerts = 64;
constexpr uint32_t kMaxRing = 128;
constexpr float kTangentFrameAreaEpsilon = 1e-10f;
constexpr float kTangentFrameDeterminantEpsilon = 1e-20f;

struct EdgeKey {
  uint32_t tail = 0;
  uint32_t head = 0;

  bool operator==(const EdgeKey& other) const {
    return tail == other.tail && head == other.head;
  }
};

struct EdgeKeyHash {
  size_t operator()(const EdgeKey& key) const {
    size_t hash = std::hash<uint32_t>{}(key.tail);
    hash ^= std::hash<uint32_t>{}(key.head) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

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

bool is_finite(Vec3 v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
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

void build_halfedges(
    uint32_t vertex_count,
    const std::vector<std::vector<uint32_t>>& faces,
    PoseTrainerCache& cache) {
  uint32_t total_halfedges = 0;
  for (const auto& face : faces) {
    if (face.size() < 3) {
      continue;
    }
    if (face.size() > kMaxFaceVerts) {
      throw std::invalid_argument("faces with more than 64 vertices are not supported by the Mush3D half-edge relax shader");
    }
    total_halfedges += static_cast<uint32_t>(face.size());
  }

  cache.halfedge_twin.assign(total_halfedges, kBoundaryHalfedge);
  cache.halfedge_next.assign(total_halfedges, kBoundaryHalfedge);
  cache.halfedge_vertex.assign(total_halfedges, kBoundaryHalfedge);
  cache.vertex_halfedge.assign(vertex_count, kBoundaryHalfedge);

  std::vector<uint32_t> halfedge_tail(total_halfedges, kBoundaryHalfedge);
  std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edge_map;
  edge_map.reserve(total_halfedges);

  uint32_t he_index = 0;
  for (const auto& face : faces) {
    const uint32_t count = static_cast<uint32_t>(face.size());
    if (count < 3) {
      continue;
    }

    const uint32_t first_he = he_index;
    for (uint32_t e = 0; e < count; ++e) {
      const uint32_t tail = face[e];
      const uint32_t head = face[(e + 1) % count];
      if (tail >= vertex_count || head >= vertex_count) {
        throw std::invalid_argument("face index out of range");
      }

      halfedge_tail[he_index] = tail;
      cache.halfedge_vertex[he_index] = head;
      cache.halfedge_next[he_index] = first_he + ((e + 1) % count);

      const EdgeKey key{tail, head};
      if (edge_map.find(key) == edge_map.end()) {
        edge_map.emplace(key, he_index);
      }

      if (cache.vertex_halfedge[tail] == kBoundaryHalfedge) {
        cache.vertex_halfedge[tail] = he_index;
      }
      he_index++;
    }
  }

  for (const auto& item : edge_map) {
    const EdgeKey reverse{item.first.head, item.first.tail};
    const auto reverse_it = edge_map.find(reverse);
    if (reverse_it != edge_map.end()) {
      cache.halfedge_twin[item.second] = reverse_it->second;
    }
  }

  for (uint32_t he = 0; he < total_halfedges; ++he) {
    if (cache.halfedge_twin[he] == kBoundaryHalfedge) {
      cache.vertex_halfedge[halfedge_tail[he]] = he;
    }
  }
}

uint32_t halfedge_prev_of(const std::vector<uint32_t>& halfedge_next, uint32_t he) {
  uint32_t previous = he;
  for (uint32_t i = 0; i < kMaxFaceVerts; ++i) {
    const uint32_t next = halfedge_next[previous];
    if (next == he) {
      return previous;
    }
    previous = next;
  }
  return he;
}

template <typename AddNeighbor>
void visit_mush3d_halfedge_ring(
    const std::vector<uint32_t>& halfedge_twin,
    const std::vector<uint32_t>& halfedge_next,
    const std::vector<uint32_t>& halfedge_vertex,
    const std::vector<uint32_t>& vertex_halfedge,
    uint32_t vertex,
    AddNeighbor&& add_neighbor) {
  const uint32_t start_he = vertex_halfedge[vertex];
  if (start_he == kBoundaryHalfedge) {
    return;
  }

  const bool start_is_boundary = halfedge_twin[start_he] == kBoundaryHalfedge;
  if (!start_is_boundary) {
    uint32_t he = start_he;
    for (uint32_t i = 0; i < kMaxRing; ++i) {
      add_neighbor(halfedge_vertex[he]);
      const uint32_t twin = halfedge_twin[he];
      if (twin == kBoundaryHalfedge) {
        break;
      }
      he = halfedge_next[twin];
      if (he == start_he) {
        break;
      }
    }
    return;
  }

  add_neighbor(halfedge_vertex[start_he]);
  uint32_t previous = halfedge_prev_of(halfedge_next, start_he);
  for (uint32_t i = 0; i < kMaxRing; ++i) {
    const uint32_t twin = halfedge_twin[previous];
    if (twin == kBoundaryHalfedge) {
      add_neighbor(halfedge_vertex[halfedge_prev_of(halfedge_next, previous)]);
      break;
    }
    add_neighbor(halfedge_vertex[twin]);
    previous = halfedge_prev_of(halfedge_next, twin);
  }
}

std::pair<std::vector<uint32_t>, bool> ordered_halfedge_ring(
    const std::vector<uint32_t>& halfedge_twin,
    const std::vector<uint32_t>& halfedge_next,
    const std::vector<uint32_t>& halfedge_vertex,
    const std::vector<uint32_t>& vertex_halfedge,
    uint32_t vertex) {
  std::vector<uint32_t> ring;
  const uint32_t start_he = vertex_halfedge[vertex];
  if (start_he == kBoundaryHalfedge) {
    return {ring, false};
  }

  const bool start_is_boundary = halfedge_twin[start_he] == kBoundaryHalfedge;
  if (!start_is_boundary) {
    uint32_t he = start_he;
    for (uint32_t i = 0; i < kMaxRing; ++i) {
      ring.push_back(halfedge_vertex[he]);
      const uint32_t twin = halfedge_twin[he];
      if (twin == kBoundaryHalfedge) {
        break;
      }
      he = halfedge_next[twin];
      if (he == start_he) {
        break;
      }
    }
    return {ring, false};
  }

  ring.push_back(halfedge_vertex[start_he]);
  uint32_t previous = halfedge_prev_of(halfedge_next, start_he);
  for (uint32_t i = 0; i < kMaxRing; ++i) {
    const uint32_t twin = halfedge_twin[previous];
    if (twin == kBoundaryHalfedge) {
      ring.push_back(halfedge_vertex[halfedge_prev_of(halfedge_next, previous)]);
      break;
    }
    ring.push_back(halfedge_vertex[twin]);
    previous = halfedge_prev_of(halfedge_next, twin);
  }
  return {ring, true};
}

void add_triangle_frame(
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& frames_by_vertex,
    uint32_t center,
    uint32_t neighbor_a,
    uint32_t neighbor_b) {
  if (center >= frames_by_vertex.size() ||
      neighbor_a >= frames_by_vertex.size() ||
      neighbor_b >= frames_by_vertex.size()) {
    throw std::invalid_argument("face index out of range");
  }
  frames_by_vertex[center].push_back({neighbor_a, neighbor_b});
}

void build_tangent_frame_table(
    PoseTrainerCache& cache,
    const std::vector<std::vector<uint32_t>>& faces) {
  cache.tangent_frame_offsets.assign(cache.vertex_count + 1, 0);
  cache.tangent_frame_center_indices.clear();
  cache.tangent_frame_neighbor_a.clear();
  cache.tangent_frame_neighbor_b.clear();

  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> frames_by_vertex(cache.vertex_count);
  for (const auto& face : faces) {
    if (face.size() < 3) {
      continue;
    }
    for (size_t i = 1; i + 1 < face.size(); ++i) {
      const uint32_t a = face[0];
      const uint32_t b = face[i];
      const uint32_t c = face[i + 1];
      add_triangle_frame(frames_by_vertex, a, b, c);
      add_triangle_frame(frames_by_vertex, b, c, a);
      add_triangle_frame(frames_by_vertex, c, a, b);
    }
  }

  for (uint32_t v = 0; v < cache.vertex_count; ++v) {
    for (const auto& frame : frames_by_vertex[v]) {
      cache.tangent_frame_center_indices.push_back(v);
      cache.tangent_frame_neighbor_a.push_back(frame.first);
      cache.tangent_frame_neighbor_b.push_back(frame.second);
    }
    cache.tangent_frame_offsets[v + 1] = static_cast<uint32_t>(cache.tangent_frame_center_indices.size());
  }
}

bool tangent_basis(
    const std::vector<Vec3>& points,
    uint32_t center,
    uint32_t neighbor_a,
    uint32_t neighbor_b,
    Matrix3f& out) {
  const Vec3 p1 = points[center];
  const Vec3 p2 = points[neighbor_a];
  const Vec3 p3 = points[neighbor_b];
  const Vector3f e1_raw = to_eigen(p2 - p1);
  const Vector3f e2_raw = to_eigen(p3 - p1);
  if (!e1_raw.allFinite() || !e2_raw.allFinite()) {
    out = Matrix3f::Identity();
    return false;
  }
  const Vector3f normal = e1_raw.cross(e2_raw);
  const float area2 = normal.norm();
  if (!std::isfinite(area2) || area2 < kTangentFrameAreaEpsilon || !normal.allFinite()) {
    out = Matrix3f::Identity();
    return false;
  }

  out.col(0) = e1_raw;
  out.col(1) = e2_raw;
  out.col(2) = normal;
  return true;
}

void pack_column_major_matrix(const Matrix3f& matrix, std::vector<float>& values, size_t base) {
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      values[base + static_cast<size_t>(col) * 3 + row] = matrix(row, col);
    }
  }
}

void compute_tangent_frames_cpu(
    const PoseTrainerCache& cache,
    const std::vector<Vec3>& points,
    std::vector<float>& frames,
    std::vector<uint32_t>& valid) {
  const size_t frame_count = cache.tangent_frame_center_indices.size();
  frames.assign(frame_count * 9, 0.0f);
  valid.assign(frame_count, 0);
  for (size_t frame = 0; frame < frame_count; ++frame) {
    Matrix3f basis_matrix;
    if (!tangent_basis(
            points,
            cache.tangent_frame_center_indices[frame],
            cache.tangent_frame_neighbor_a[frame],
            cache.tangent_frame_neighbor_b[frame],
            basis_matrix)) {
      continue;
    }
    valid[frame] = 1;
    pack_column_major_matrix(basis_matrix, frames, frame * 9);
  }
}

void pack_sample_base_frames(PoseTrainerCache& cache) {
  const size_t shape_count = cache.sample_relaxed.size();
  const size_t frame_count = cache.tangent_frame_center_indices.size();
  cache.sample_base_frame_inverse_flat.assign(shape_count * frame_count * 9, 0.0f);
  cache.sample_base_frame_valid_flat.assign(shape_count * frame_count, 0);

  for (size_t s = 0; s < shape_count; ++s) {
    for (size_t frame = 0; frame < frame_count; ++frame) {
      Matrix3f basis_matrix;
      if (!tangent_basis(
              cache.sample_relaxed[s],
              cache.tangent_frame_center_indices[frame],
              cache.tangent_frame_neighbor_a[frame],
              cache.tangent_frame_neighbor_b[frame],
              basis_matrix)) {
        continue;
      }
      const float determinant = basis_matrix.determinant();
      if (!std::isfinite(determinant) || std::abs(determinant) < kTangentFrameDeterminantEpsilon) {
        continue;
      }
      const Matrix3f inverse_matrix = basis_matrix.inverse();
      if (!inverse_matrix.allFinite()) {
        continue;
      }
      cache.sample_base_frame_valid_flat[s * frame_count + frame] = 1;
      pack_column_major_matrix(
          inverse_matrix,
          cache.sample_base_frame_inverse_flat,
          (s * frame_count + frame) * 9);
    }
  }
}

Vec3 mul_column_major_matrix(const std::vector<float>& values, size_t base, Vec3 v) {
  return {
      values[base] * v.x + values[base + 3] * v.y + values[base + 6] * v.z,
      values[base + 1] * v.x + values[base + 4] * v.y + values[base + 7] * v.z,
      values[base + 2] * v.x + values[base + 5] * v.y + values[base + 8] * v.z};
}

Vec3 transport_sample_delta(
    const PoseTrainerCache& cache,
    const std::vector<float>& current_frames,
    const std::vector<uint32_t>& current_frame_valid,
    size_t sample,
    uint32_t vertex,
    Vec3 delta) {
  const float delta_len = length(delta);
  if (delta_len < 0.001f) {
    return delta;
  }

  const size_t frame_count = cache.tangent_frame_center_indices.size();
  const uint32_t start = cache.tangent_frame_offsets[vertex];
  const uint32_t end = cache.tangent_frame_offsets[vertex + 1];
  Vec3 sum;
  uint32_t valid_count = 0;
  for (uint32_t frame = start; frame < end; ++frame) {
    if (current_frame_valid[frame] == 0 ||
        cache.sample_base_frame_valid_flat[sample * frame_count + frame] == 0) {
      continue;
    }
    const Vec3 local_delta = mul_column_major_matrix(
        cache.sample_base_frame_inverse_flat,
        (sample * frame_count + frame) * 9,
        delta);
    if (!is_finite(local_delta)) {
      continue;
    }
    const Vec3 rotated = mul_column_major_matrix(current_frames, static_cast<size_t>(frame) * 9, local_delta);
    if (!is_finite(rotated)) {
      continue;
    }
    sum += rotated;
    valid_count++;
  }
  if (valid_count == 0) {
    return delta;
  }
  if (!is_finite(sum)) {
    return delta;
  }
  Vec3 average = sum / static_cast<float>(valid_count);
  const float average_len = length(average);
  if (std::isfinite(average_len) && average_len > 1e-8f) {
    average = average * (delta_len / average_len);
  }
  return average;
}

std::vector<Vec3> relax(
    const std::vector<Vec3>& points,
    const std::vector<uint32_t>& halfedge_twin,
    const std::vector<uint32_t>& halfedge_next,
    const std::vector<uint32_t>& halfedge_vertex,
    const std::vector<uint32_t>& vertex_halfedge,
    int iterations) {
  std::vector<Vec3> current = points;
  for (int iter = 0; iter < std::max(0, iterations); ++iter) {
    std::vector<Vec3> next = current;
    for (size_t v = 0; v < current.size(); ++v) {
      Vec3 center;
      uint32_t count = 0;
      visit_mush3d_halfedge_ring(
          halfedge_twin,
          halfedge_next,
          halfedge_vertex,
          vertex_halfedge,
          static_cast<uint32_t>(v),
          [&](uint32_t neighbor) {
            center += current[neighbor];
            count++;
          });
      if (count == 0) {
        continue;
      }
      next[v] = center / static_cast<float>(count);
    }
    current.swap(next);
  }
  return current;
}

std::vector<float> relax_weights(
    const std::vector<float>& weights,
    const std::vector<uint32_t>& halfedge_twin,
    const std::vector<uint32_t>& halfedge_next,
    const std::vector<uint32_t>& halfedge_vertex,
    const std::vector<uint32_t>& vertex_halfedge,
    int iterations) {
  std::vector<float> current = weights;
  for (int iter = 0; iter < std::max(0, iterations); ++iter) {
    std::vector<float> next = current;
    for (size_t v = 0; v < current.size(); ++v) {
      float total = 0.0f;
      uint32_t count = 0;
      visit_mush3d_halfedge_ring(
          halfedge_twin,
          halfedge_next,
          halfedge_vertex,
          vertex_halfedge,
          static_cast<uint32_t>(v),
          [&](uint32_t neighbor) {
            total += current[neighbor];
            count++;
          });
      if (count == 0) {
        continue;
      }
      next[v] = total / static_cast<float>(count);
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

void train_area_model_cpu(PoseTrainerCache& cache, AreaModel& model, int sample_count) {
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
  model.theta = train_theta(model.features, sample_count, cache.settings.rbf_radius, cache.settings.regularization);
}

std::vector<float> build_opencl_train_rep_positions(const PoseTrainerCache& cache, int sample_count) {
  std::vector<float> rep_positions(
      cache.areas.size() * sample_count * kRepresentativeVertexCount * 3,
      0.0f);
  for (const AreaModel& area : cache.areas) {
    const size_t area_id = area.area_id;
    for (int r = 0; r < kRepresentativeVertexCount; ++r) {
      const uint32_t rep = area.reps[r];
      const Vec3 bind_p = cache.bind_relaxed[rep];
      size_t base = ((area_id * sample_count) * kRepresentativeVertexCount + r) * 3;
      rep_positions[base] = bind_p.x;
      rep_positions[base + 1] = bind_p.y;
      rep_positions[base + 2] = bind_p.z;
      for (size_t s = 0; s < cache.sample_relaxed.size(); ++s) {
        const Vec3 sample_p = cache.sample_relaxed[s][rep];
        base = ((area_id * sample_count + s + 1) * kRepresentativeVertexCount + r) * 3;
        rep_positions[base] = sample_p.x;
        rep_positions[base + 1] = sample_p.y;
        rep_positions[base + 2] = sample_p.z;
      }
    }
  }
  return rep_positions;
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

void pack_neighbor_csr(PoseTrainerCache& cache) {
  cache.neighbor_offsets.assign(cache.vertex_count + 1, 0);
  for (uint32_t v = 0; v < cache.vertex_count; ++v) {
    cache.neighbor_offsets[v + 1] = cache.neighbor_offsets[v] + static_cast<uint32_t>(cache.neighbors[v].size());
  }
  cache.neighbor_indices.resize(cache.neighbor_offsets.back());
  for (uint32_t v = 0; v < cache.vertex_count; ++v) {
    std::copy(
        cache.neighbors[v].begin(),
        cache.neighbors[v].end(),
        cache.neighbor_indices.begin() + cache.neighbor_offsets[v]);
  }
}

void pack_membership_csr(PoseTrainerCache& cache) {
  cache.membership_offsets.assign(cache.vertex_count + 1, 0);
  for (const AreaModel& area : cache.areas) {
    for (uint32_t v : area.active_vertices) {
      cache.membership_offsets[v + 1] += 1;
    }
  }
  for (uint32_t v = 0; v < cache.vertex_count; ++v) {
    cache.membership_offsets[v + 1] += cache.membership_offsets[v];
  }

  const uint32_t nnz = cache.membership_offsets.back();
  cache.membership_area_ids.assign(nnz, 0);
  cache.membership_weights.assign(nnz, 0.0f);
  std::vector<uint32_t> cursor = cache.membership_offsets;
  for (const AreaModel& area : cache.areas) {
    for (size_t i = 0; i < area.active_vertices.size(); ++i) {
      const uint32_t v = area.active_vertices[i];
      const uint32_t dst = cursor[v]++;
      cache.membership_area_ids[dst] = area.area_id;
      cache.membership_weights[dst] = area.active_weights[i];
    }
  }
}

void pack_sample_deltas(PoseTrainerCache& cache) {
  cache.sample_deltas_flat.assign(cache.sample_deltas.size() * cache.vertex_count * 3, 0.0f);
  for (size_t s = 0; s < cache.sample_deltas.size(); ++s) {
    for (uint32_t v = 0; v < cache.vertex_count; ++v) {
      const size_t offset = (s * cache.vertex_count + v) * 3;
      const Vec3 d = cache.sample_deltas[s][v];
      cache.sample_deltas_flat[offset] = d.x;
      cache.sample_deltas_flat[offset + 1] = d.y;
      cache.sample_deltas_flat[offset + 2] = d.z;
    }
  }
}

void pack_opencl_area_data(PoseTrainerCache& cache, int sample_count) {
  const size_t area_count = cache.areas.size();
  cache.rep_indices_flat.assign(area_count * kRepresentativeVertexCount, 0);
  cache.feature_values_flat.assign(area_count * sample_count * kRepresentativeVertexCount * 3, 0.0f);
  cache.theta_values_flat.assign(area_count * sample_count * sample_count, 0.0f);
  cache.scale_values.assign(area_count, 1.0f);

  for (const AreaModel& area : cache.areas) {
    const size_t area_id = area.area_id;
    cache.scale_values[area_id] = area.scale;
    for (int r = 0; r < kRepresentativeVertexCount; ++r) {
      const uint32_t rep = area.reps[r];
      cache.rep_indices_flat[area_id * kRepresentativeVertexCount + r] = rep;
    }

    for (int s = 0; s < sample_count; ++s) {
      for (int r = 0; r < kRepresentativeVertexCount; ++r) {
        const Vec3 p = area.features[s * kRepresentativeVertexCount + r];
        const size_t base = ((area_id * sample_count + s) * kRepresentativeVertexCount + r) * 3;
        cache.feature_values_flat[base] = p.x;
        cache.feature_values_flat[base + 1] = p.y;
        cache.feature_values_flat[base + 2] = p.z;
      }
    }

    for (int row = 0; row < sample_count; ++row) {
      for (int col = 0; col < sample_count; ++col) {
        cache.theta_values_flat[(area_id * sample_count + row) * sample_count + col] =
            area.theta[row * sample_count + col];
      }
    }
  }
}

std::vector<float> evaluate_area_activations(
    const PoseTrainerCache& cache,
    const std::vector<Vec3>& relaxed) {
  const int sample_count = static_cast<int>(cache.sample_deltas.size()) + 1;
  std::vector<float> activations(cache.areas.size() * sample_count, 0.0f);

  for (const AreaModel& area : cache.areas) {
    const auto current_feature = gather_reps(relaxed, area.reps);
    const auto bind_feature = gather_reps(cache.bind_relaxed, area.reps);
    auto aligned_current = align_points(current_feature, bind_feature);
    for (Vec3& p : aligned_current) {
      p = p * area.scale;
    }

    std::vector<float> phi(sample_count, 0.0f);
    for (int s = 0; s < sample_count; ++s) {
      float total = 0.0f;
      for (int r = 0; r < kRepresentativeVertexCount; ++r) {
        total += basis(length(aligned_current[r] - area.features[s * kRepresentativeVertexCount + r]), cache.settings.rbf_radius);
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
    std::copy(
        activation.begin(),
        activation.end(),
        activations.begin() + static_cast<size_t>(area.area_id) * sample_count);
  }
  return activations;
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
  pack_neighbor_csr(cache);
  build_halfedges(cache.vertex_count, faces, cache);
  build_tangent_frame_table(cache, faces);

  // OpenCL training is still experimental; keep Auto on the proven CPU path
  // while explicit OpenCL remains available for parity and profiling.
  const bool try_opencl_training = settings.runtime_backend == 2;
  const bool require_opencl_training = settings.runtime_backend == 2;
  bool relaxed_with_opencl = false;
  if (try_opencl_training) {
    try {
      cache.bind_relaxed = relax_opencl(cache, bind, settings.relax_iterations);
      for (const auto& sample : samples) {
        std::vector<Vec3> relaxed = relax_opencl(cache, sample, settings.relax_iterations);
        std::vector<Vec3> delta(sample.size());
        for (size_t i = 0; i < sample.size(); ++i) {
          delta[i] = sample[i] - relaxed[i];
        }
        cache.sample_relaxed.push_back(std::move(relaxed));
        cache.sample_deltas.push_back(std::move(delta));
      }
      relaxed_with_opencl = true;
    } catch (const std::exception&) {
      if (require_opencl_training) {
        throw;
      }
      cache.bind_relaxed.clear();
      cache.sample_relaxed.clear();
      cache.sample_deltas.clear();
    }
  }

  if (!relaxed_with_opencl) {
    cache.bind_relaxed = relax(
        bind,
        cache.halfedge_twin,
        cache.halfedge_next,
        cache.halfedge_vertex,
        cache.vertex_halfedge,
        settings.relax_iterations);

    for (const auto& sample : samples) {
      std::vector<Vec3> relaxed = relax(
          sample,
          cache.halfedge_twin,
          cache.halfedge_next,
          cache.halfedge_vertex,
          cache.vertex_halfedge,
          settings.relax_iterations);
      std::vector<Vec3> delta(sample.size());
      for (size_t i = 0; i < sample.size(); ++i) {
        delta[i] = sample[i] - relaxed[i];
      }
      cache.sample_relaxed.push_back(std::move(relaxed));
      cache.sample_deltas.push_back(std::move(delta));
    }
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
    model.vertex_weights = relax_weights(
        input.weights,
        cache.halfedge_twin,
        cache.halfedge_next,
        cache.halfedge_vertex,
        cache.vertex_halfedge,
        settings.area_relax_iterations);
    model.reps = sample_reps(cache.bind_relaxed, model.vertex_weights);
    for (uint32_t v = 0; v < model.vertex_weights.size(); ++v) {
      const float w = model.vertex_weights[v];
      if (w > 0.0f) {
        model.active_vertices.push_back(v);
        model.active_weights.push_back(w);
      }
    }
    cache.areas.push_back(std::move(model));
  }

  bool trained_areas_with_opencl = false;
  if (try_opencl_training && !cache.areas.empty()) {
    try {
      const std::vector<float> rep_positions = build_opencl_train_rep_positions(cache, sample_count);
      train_area_models_opencl(cache, rep_positions, sample_count);
      trained_areas_with_opencl = true;
    } catch (const std::exception&) {
      if (require_opencl_training) {
        throw;
      }
    }
  }

  if (!trained_areas_with_opencl) {
    for (AreaModel& area : cache.areas) {
      train_area_model_cpu(cache, area, sample_count);
    }
  }

  cache.total_area_weights.assign(cache.vertex_count, 0.0f);
  for (const AreaModel& area : cache.areas) {
    for (size_t i = 0; i < area.active_vertices.size(); ++i) {
      cache.total_area_weights[area.active_vertices[i]] += area.active_weights[i];
    }
  }
  pack_membership_csr(cache);
  pack_sample_deltas(cache);
  pack_sample_base_frames(cache);
  pack_opencl_area_data(cache, sample_count);

  return cache;
}

std::vector<Vec3> PoseTrainerCache::evaluate(
    const std::vector<Vec3>& animated,
    const std::vector<float>& vertex_mask,
    float envelope,
    int solve_iterations_override,
    int runtime_backend_override,
    bool profile_timing) const {
  if (animated.size() != vertex_count) {
    throw std::invalid_argument("animated vertex count does not match cache");
  }
  if (!vertex_mask.empty() && vertex_mask.size() != vertex_count) {
    throw std::invalid_argument("vertex mask length does not match cache");
  }

  const int backend = runtime_backend_override >= 0 ? runtime_backend_override : settings.runtime_backend;
  const int iterations = std::max(1, solve_iterations_override > 0 ? solve_iterations_override : settings.solve_iterations);
  if (backend != 1) {
    try {
      std::vector<Vec3> out = evaluate_opencl(*this, animated, vertex_mask, envelope, iterations, profile_timing);
      last_backend = "OpenCL";
      return out;
    } catch (const std::exception&) {
      if (backend == 2) {
        throw;
      }
    }
  }
  last_backend = "CPU";
  last_opencl_timing.clear();

  std::vector<Vec3> current = animated;
  const int sample_count = static_cast<int>(sample_deltas.size()) + 1;

  for (int iter = 0; iter < iterations; ++iter) {
    const std::vector<Vec3> relaxed = relax(
        current,
        halfedge_twin,
        halfedge_next,
        halfedge_vertex,
        vertex_halfedge,
        settings.relax_iterations);
    std::vector<Vec3> pose_delta(vertex_count);
    for (uint32_t v = 0; v < vertex_count; ++v) {
      pose_delta[v] = animated[v] - relaxed[v];
    }

    std::vector<Vec3> accum(vertex_count);
    std::vector<float> current_frames;
    std::vector<uint32_t> current_frame_valid;
    compute_tangent_frames_cpu(*this, relaxed, current_frames, current_frame_valid);
    const std::vector<float> activation_flat = evaluate_area_activations(*this, relaxed);

    for (const AreaModel& area : areas) {
      for (size_t i = 0; i < area.active_vertices.size(); ++i) {
        const uint32_t v = area.active_vertices[i];
        const float area_w = area.active_weights[i];
        const size_t activation_base = static_cast<size_t>(area.area_id) * sample_count;
        accum[v] += pose_delta[v] * (area_w * activation_flat[activation_base]);
        for (size_t s = 0; s < sample_deltas.size(); ++s) {
          const Vec3 transported = transport_sample_delta(
              *this,
              current_frames,
              current_frame_valid,
              s,
              v,
              sample_deltas[s][v]);
          accum[v] += transported * (area_w * activation_flat[activation_base + s + 1]);
        }
      }
    }

    std::vector<Vec3> output(vertex_count);
    for (uint32_t v = 0; v < vertex_count; ++v) {
      const float uncovered = std::max(1.0f - total_area_weights[v], 0.0f);
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

bool opencl_is_available() {
  return opencl_runtime_available();
}

std::string opencl_status() {
  return opencl_runtime_status();
}

}  // namespace pose_trainer
