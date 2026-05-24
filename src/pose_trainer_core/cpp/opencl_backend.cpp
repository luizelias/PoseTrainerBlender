#include "opencl_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pose_trainer {
namespace {

using cl_bool = uint32_t;
using cl_bitfield = uint64_t;
using cl_device_type = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_mem = struct _cl_mem*;
using cl_event = struct _cl_event*;
using cl_context_properties = intptr_t;
using cl_profiling_info = cl_uint;
using cl_ulong = uint64_t;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bool CL_FALSE_VALUE = 0;
constexpr cl_bool CL_TRUE_VALUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU = 1ull << 2;
constexpr cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFFull;
constexpr cl_bitfield CL_QUEUE_PROFILING_ENABLE = 1ull << 1;
constexpr cl_mem_flags CL_MEM_READ_WRITE = 1ull << 0;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_profiling_info CL_PROFILING_COMMAND_START = 0x1282;
constexpr cl_profiling_info CL_PROFILING_COMMAND_END = 0x1283;

struct OpenClApi {
#ifdef _WIN32
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  std::string status = "OpenCL not initialized";

  using FnGetPlatformIDs = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
  using FnGetDeviceIDs = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
  using FnCreateContext = cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
  using FnCreateCommandQueue = cl_command_queue (*)(cl_context, cl_device_id, cl_bitfield, cl_int*);
  using FnCreateProgramWithSource = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
  using FnBuildProgram = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
  using FnGetProgramBuildInfo = cl_int (*)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
  using FnCreateKernel = cl_kernel (*)(cl_program, const char*, cl_int*);
  using FnCreateBuffer = cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
  using FnSetKernelArg = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
  using FnEnqueueWriteBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const void*, void*);
  using FnEnqueueReadBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const void*, void*);
  using FnEnqueueNDRangeKernel = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
  using FnFinish = cl_int (*)(cl_command_queue);
  using FnGetEventProfilingInfo = cl_int (*)(cl_event, cl_profiling_info, size_t, void*, size_t*);
  using FnReleaseEvent = cl_int (*)(cl_event);
  using FnReleaseMemObject = cl_int (*)(cl_mem);
  using FnReleaseKernel = cl_int (*)(cl_kernel);
  using FnReleaseProgram = cl_int (*)(cl_program);
  using FnReleaseCommandQueue = cl_int (*)(cl_command_queue);
  using FnReleaseContext = cl_int (*)(cl_context);

  FnGetPlatformIDs clGetPlatformIDs = nullptr;
  FnGetDeviceIDs clGetDeviceIDs = nullptr;
  FnCreateContext clCreateContext = nullptr;
  FnCreateCommandQueue clCreateCommandQueue = nullptr;
  FnCreateProgramWithSource clCreateProgramWithSource = nullptr;
  FnBuildProgram clBuildProgram = nullptr;
  FnGetProgramBuildInfo clGetProgramBuildInfo = nullptr;
  FnCreateKernel clCreateKernel = nullptr;
  FnCreateBuffer clCreateBuffer = nullptr;
  FnSetKernelArg clSetKernelArg = nullptr;
  FnEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
  FnEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
  FnEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
  FnFinish clFinish = nullptr;
  FnGetEventProfilingInfo clGetEventProfilingInfo = nullptr;
  FnReleaseEvent clReleaseEvent = nullptr;
  FnReleaseMemObject clReleaseMemObject = nullptr;
  FnReleaseKernel clReleaseKernel = nullptr;
  FnReleaseProgram clReleaseProgram = nullptr;
  FnReleaseCommandQueue clReleaseCommandQueue = nullptr;
  FnReleaseContext clReleaseContext = nullptr;

  OpenClApi() {
#ifdef _WIN32
    library = LoadLibraryA("OpenCL.dll");
#else
    library = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) {
      library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    }
#endif
    if (!library) {
      status = "OpenCL library was not found";
      return;
    }

    auto load = [&](auto& fn, const char* name) {
#ifdef _WIN32
      fn = reinterpret_cast<std::decay_t<decltype(fn)>>(GetProcAddress(library, name));
#else
      fn = reinterpret_cast<std::decay_t<decltype(fn)>>(dlsym(library, name));
#endif
      return fn != nullptr;
    };

    if (!load(clGetPlatformIDs, "clGetPlatformIDs") ||
        !load(clGetDeviceIDs, "clGetDeviceIDs") ||
        !load(clCreateContext, "clCreateContext") ||
        !load(clCreateCommandQueue, "clCreateCommandQueue") ||
        !load(clCreateProgramWithSource, "clCreateProgramWithSource") ||
        !load(clBuildProgram, "clBuildProgram") ||
        !load(clGetProgramBuildInfo, "clGetProgramBuildInfo") ||
        !load(clCreateKernel, "clCreateKernel") ||
        !load(clCreateBuffer, "clCreateBuffer") ||
        !load(clSetKernelArg, "clSetKernelArg") ||
        !load(clEnqueueWriteBuffer, "clEnqueueWriteBuffer") ||
        !load(clEnqueueReadBuffer, "clEnqueueReadBuffer") ||
        !load(clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel") ||
        !load(clFinish, "clFinish") ||
        !load(clGetEventProfilingInfo, "clGetEventProfilingInfo") ||
        !load(clReleaseEvent, "clReleaseEvent") ||
        !load(clReleaseMemObject, "clReleaseMemObject") ||
        !load(clReleaseKernel, "clReleaseKernel") ||
        !load(clReleaseProgram, "clReleaseProgram") ||
        !load(clReleaseCommandQueue, "clReleaseCommandQueue") ||
        !load(clReleaseContext, "clReleaseContext")) {
      status = "OpenCL library is missing required 1.2 symbols";
      return;
    }
    status = "OpenCL library loaded";
  }

  bool loaded() const {
    return clGetPlatformIDs != nullptr;
  }
};

OpenClApi& api() {
  static OpenClApi instance;
  return instance;
}

void check(cl_int err, const char* what) {
  if (err != CL_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed with OpenCL error " + std::to_string(err));
  }
}

uint64_t hash_float_vector(const std::vector<float>& values) {
  constexpr uint64_t kOffset = 1469598103934665603ull;
  constexpr uint64_t kPrime = 1099511628211ull;
  uint64_t hash = kOffset;
  auto mix = [&](uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      hash ^= (value >> (i * 8)) & 0xffu;
      hash *= kPrime;
    }
  };
  mix(values.size());
  for (float value : values) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    mix(bits);
  }
  return hash;
}

uint64_t hash_uint_vector(const std::vector<uint32_t>& values) {
  constexpr uint64_t kOffset = 1469598103934665603ull;
  constexpr uint64_t kPrime = 1099511628211ull;
  uint64_t hash = kOffset;
  auto mix = [&](uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      hash ^= (value >> (i * 8)) & 0xffu;
      hash *= kPrime;
    }
  };
  mix(values.size());
  for (uint32_t value : values) {
    mix(value);
  }
  return hash;
}

void hash_combine(uint64_t& seed, uint64_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

std::string kernel_source() {
  return std::string(R"CLC(
__kernel void relax_neighbors(
    __global const float* in_pos,
    __global float* out_pos,
    __global const uint* he_twin,
    __global const uint* he_next,
    __global const uint* he_vertex,
    __global const uint* vertex_he,
    const uint vertex_count) {
  const uint v = get_global_id(0);
  if (v >= vertex_count) { return; }
  const uint base = v * 3u;

  const uint BOUNDARY = 0xFFFFFFFFu;
  const uint MAX_FACE_VERTS = 64u;
  const uint MAX_RING = 128u;
  const uint start_he = vertex_he[v];
  if (start_he == BOUNDARY) {
    out_pos[base] = in_pos[base];
    out_pos[base + 1u] = in_pos[base + 1u];
    out_pos[base + 2u] = in_pos[base + 2u];
    return;
  }

  float3 center = (float3)(0.0f, 0.0f, 0.0f);
  uint count = 0u;
  const uint start_is_boundary = he_twin[start_he] == BOUNDARY ? 1u : 0u;

  if (start_is_boundary == 0u) {
    uint he = start_he;
    for (uint i = 0u; i < MAX_RING; i++) {
      const uint n = he_vertex[he] * 3u;
      center += (float3)(in_pos[n], in_pos[n + 1u], in_pos[n + 2u]);
      count++;
      const uint twin = he_twin[he];
      if (twin == BOUNDARY) { break; }
      he = he_next[twin];
      if (he == start_he) { break; }
    }
  } else {
    const uint n0 = he_vertex[start_he] * 3u;
    center += (float3)(in_pos[n0], in_pos[n0 + 1u], in_pos[n0 + 2u]);
    count++;

    uint previous = start_he;
    for (uint i = 0u; i < MAX_FACE_VERTS; i++) {
      const uint next = he_next[previous];
      if (next == start_he) { break; }
      previous = next;
    }

    for (uint i = 0u; i < MAX_RING; i++) {
      const uint twin = he_twin[previous];
      if (twin == BOUNDARY) {
        uint prev_previous = previous;
        for (uint j = 0u; j < MAX_FACE_VERTS; j++) {
          const uint next = he_next[prev_previous];
          if (next == previous) { break; }
          prev_previous = next;
        }
        const uint n = he_vertex[prev_previous] * 3u;
        center += (float3)(in_pos[n], in_pos[n + 1u], in_pos[n + 2u]);
        count++;
        break;
      }
      const uint n = he_vertex[twin] * 3u;
      center += (float3)(in_pos[n], in_pos[n + 1u], in_pos[n + 2u]);
      count++;

      previous = twin;
      for (uint j = 0u; j < MAX_FACE_VERTS; j++) {
        const uint next = he_next[previous];
        if (next == twin) { break; }
        previous = next;
      }
    }
  }

  if (count == 0u) {
    out_pos[base] = in_pos[base];
    out_pos[base + 1u] = in_pos[base + 1u];
    out_pos[base + 2u] = in_pos[base + 2u];
    return;
  }

  center *= 1.0f / (float)count;
  out_pos[base] = center.x;
  out_pos[base + 1u] = center.y;
  out_pos[base + 2u] = center.z;
}

float3 read_point(__global const float* values, uint idx) {
  const uint b = idx * 3u;
  return (float3)(values[b], values[b + 1u], values[b + 2u]);
}

float3 read_feature(__global const float* features, uint area, uint sample, uint rep, uint sample_count) {
  const uint b = ((area * sample_count + sample) * 16u + rep) * 3u;
  return (float3)(features[b], features[b + 1u], features[b + 2u]);
}

void identity3(__private float* R) {
  R[0] = 1.0f; R[1] = 0.0f; R[2] = 0.0f;
  R[3] = 0.0f; R[4] = 1.0f; R[5] = 0.0f;
  R[6] = 0.0f; R[7] = 0.0f; R[8] = 1.0f;
}

float3 rotate3(__private const float* R, float3 p) {
  return (float3)(
      R[0] * p.x + R[1] * p.y + R[2] * p.z,
      R[3] * p.x + R[4] * p.y + R[5] * p.z,
      R[6] * p.x + R[7] * p.y + R[8] * p.z);
}

void quat_to_matrix(float q0, float q1, float q2, float q3, __private float* R) {
  const float xx = q1 * q1;
  const float yy = q2 * q2;
  const float zz = q3 * q3;
  const float xy = q1 * q2;
  const float xz = q1 * q3;
  const float yz = q2 * q3;
  const float wx = q0 * q1;
  const float wy = q0 * q2;
  const float wz = q0 * q3;
  R[0] = 1.0f - 2.0f * (yy + zz);
  R[1] = 2.0f * (xy - wz);
  R[2] = 2.0f * (xz + wy);
  R[3] = 2.0f * (xy + wz);
  R[4] = 1.0f - 2.0f * (xx + zz);
  R[5] = 2.0f * (yz - wx);
  R[6] = 2.0f * (xz - wy);
  R[7] = 2.0f * (yz + wx);
  R[8] = 1.0f - 2.0f * (xx + yy);
}

void procrustes_rotation16(__private const float3* src, __private const float3* dst, __private float* R) {
  float3 cs = (float3)(0.0f, 0.0f, 0.0f);
  float3 cd = (float3)(0.0f, 0.0f, 0.0f);
  for (uint i = 0u; i < 16u; i++) {
    cs += src[i];
    cd += dst[i];
  }
  cs *= 1.0f / 16.0f;
  cd *= 1.0f / 16.0f;

  float h00 = 0.0f; float h01 = 0.0f; float h02 = 0.0f;
  float h10 = 0.0f; float h11 = 0.0f; float h12 = 0.0f;
  float h20 = 0.0f; float h21 = 0.0f; float h22 = 0.0f;
  for (uint i = 0u; i < 16u; i++) {
    const float3 s = src[i] - cs;
    const float3 t = dst[i] - cd;
    h00 += s.x * t.x; h01 += s.x * t.y; h02 += s.x * t.z;
    h10 += s.y * t.x; h11 += s.y * t.y; h12 += s.y * t.z;
    h20 += s.z * t.x; h21 += s.z * t.y; h22 += s.z * t.z;
  }

  const float hnorm =
      h00*h00 + h01*h01 + h02*h02 +
      h10*h10 + h11*h11 + h12*h12 +
      h20*h20 + h21*h21 + h22*h22;
  float same_error = 0.0f;
  for (uint i = 0u; i < 16u; i++) {
    const float3 d = src[i] - dst[i];
    same_error += dot(d, d);
  }
  if (same_error < 1.0e-8f) {
    identity3(R);
    return;
  }
  if (hnorm < 1.0e-20f) {
    identity3(R);
    return;
  }

  float3 W[3];
  W[0] = (float3)(h00, h10, h20);
  W[1] = (float3)(h01, h11, h21);
  W[2] = (float3)(h02, h12, h22);
  float3 V[3];
  V[0] = (float3)(1.0f, 0.0f, 0.0f);
  V[1] = (float3)(0.0f, 1.0f, 0.0f);
  V[2] = (float3)(0.0f, 0.0f, 1.0f);

  for (uint sweep = 0u; sweep < 12u; sweep++) {
    for (uint pair = 0u; pair < 3u; pair++) {
      const uint p = pair == 0u ? 0u : (pair == 1u ? 0u : 1u);
      const uint q = pair == 0u ? 1u : (pair == 1u ? 2u : 2u);
      const float3 col_p = W[p];
      const float3 col_q = W[q];
      const float alpha = dot(col_p, col_p);
      const float beta = dot(col_q, col_q);
      const float gamma = dot(col_p, col_q);
      if (fabs(gamma) < 1.0e-10f * sqrt(alpha * beta + 1.0e-20f)) {
        continue;
      }
      const float diff = beta - alpha;
      float t;
      if (fabs(diff) < 1.0e-10f) {
        t = 1.0f;
      } else {
        const float zeta = diff / (2.0f * gamma);
        t = copysign(1.0f, zeta) / (fabs(zeta) + sqrt(1.0f + zeta * zeta));
      }
      const float c = 1.0f / sqrt(1.0f + t * t);
      const float s = t * c;
      W[p] = c * col_p - s * col_q;
      W[q] = s * col_p + c * col_q;
      const float3 vp = V[p];
      const float3 vq = V[q];
      V[p] = c * vp - s * vq;
      V[q] = s * vp + c * vq;
    }
  }

  float sigma[3];
  sigma[0] = length(W[0]);
  sigma[1] = length(W[1]);
  sigma[2] = length(W[2]);
  uint idx[3] = {0u, 1u, 2u};
  if (sigma[idx[0]] < sigma[idx[1]]) { uint tmp = idx[0]; idx[0] = idx[1]; idx[1] = tmp; }
  if (sigma[idx[1]] < sigma[idx[2]]) { uint tmp = idx[1]; idx[1] = idx[2]; idx[2] = tmp; }
  if (sigma[idx[0]] < sigma[idx[1]]) { uint tmp = idx[0]; idx[0] = idx[1]; idx[1] = tmp; }

  float3 U[3];
  float3 VS[3];
  for (uint i = 0u; i < 3u; i++) {
    const uint k = idx[i];
    VS[i] = V[k];
    if (sigma[k] > 1.0e-7f) {
      U[i] = W[k] / sigma[k];
    } else if (i == 0u) {
      U[i] = (float3)(1.0f, 0.0f, 0.0f);
    } else if (i == 1u) {
      const float3 axis = fabs(U[0].x) < 0.9f ? (float3)(1.0f, 0.0f, 0.0f) : (float3)(0.0f, 1.0f, 0.0f);
      U[i] = normalize(cross(U[0], axis));
    } else {
      U[i] = normalize(cross(U[0], U[1]));
    }
  }
  U[2] = normalize(cross(U[0], U[1]));

  R[0] = VS[0].x * U[0].x + VS[1].x * U[1].x + VS[2].x * U[2].x;
  R[1] = VS[0].x * U[0].y + VS[1].x * U[1].y + VS[2].x * U[2].y;
  R[2] = VS[0].x * U[0].z + VS[1].x * U[1].z + VS[2].x * U[2].z;
  R[3] = VS[0].y * U[0].x + VS[1].y * U[1].x + VS[2].y * U[2].x;
  R[4] = VS[0].y * U[0].y + VS[1].y * U[1].y + VS[2].y * U[2].y;
  R[5] = VS[0].y * U[0].z + VS[1].y * U[1].z + VS[2].y * U[2].z;
  R[6] = VS[0].z * U[0].x + VS[1].z * U[1].x + VS[2].z * U[2].x;
  R[7] = VS[0].z * U[0].y + VS[1].z * U[1].y + VS[2].z * U[2].y;
  R[8] = VS[0].z * U[0].z + VS[1].z * U[1].z + VS[2].z * U[2].z;
  const float det =
      R[0] * (R[4] * R[8] - R[5] * R[7]) -
      R[1] * (R[3] * R[8] - R[5] * R[6]) +
      R[2] * (R[3] * R[7] - R[4] * R[6]);
  if (det < 0.0f) {
    VS[2] = -VS[2];
    R[0] = VS[0].x * U[0].x + VS[1].x * U[1].x + VS[2].x * U[2].x;
    R[1] = VS[0].x * U[0].y + VS[1].x * U[1].y + VS[2].x * U[2].y;
    R[2] = VS[0].x * U[0].z + VS[1].x * U[1].z + VS[2].x * U[2].z;
    R[3] = VS[0].y * U[0].x + VS[1].y * U[1].x + VS[2].y * U[2].x;
    R[4] = VS[0].y * U[0].y + VS[1].y * U[1].y + VS[2].y * U[2].y;
    R[5] = VS[0].y * U[0].z + VS[1].y * U[1].z + VS[2].y * U[2].z;
    R[6] = VS[0].z * U[0].x + VS[1].z * U[1].x + VS[2].z * U[2].x;
    R[7] = VS[0].z * U[0].y + VS[1].z * U[1].y + VS[2].z * U[2].y;
    R[8] = VS[0].z * U[0].z + VS[1].z * U[1].z + VS[2].z * U[2].z;
  }
}

void project_simplex_private(__private float* w, uint n) {
  float sorted[32];
  for (uint i = 0u; i < n; i++) {
    sorted[i] = w[i];
  }
  for (uint i = 0u; i < n; i++) {
    for (uint j = i + 1u; j < n; j++) {
      if (sorted[j] > sorted[i]) {
        const float tmp = sorted[i];
        sorted[i] = sorted[j];
        sorted[j] = tmp;
      }
    }
  }
  float cumulative = 0.0f;
  uint rho = 0u;
  for (uint i = 0u; i < n; i++) {
    cumulative += sorted[i];
    const float threshold = (cumulative - 1.0f) / (float)(i + 1u);
    if (sorted[i] - threshold > 0.0f) {
      rho = i;
    }
  }
  cumulative = 0.0f;
  for (uint i = 0u; i <= rho; i++) {
    cumulative += sorted[i];
  }
  const float tau = (cumulative - 1.0f) / (float)(rho + 1u);
  for (uint i = 0u; i < n; i++) {
    w[i] = fmax(w[i] - tau, 0.0f);
  }
}
)CLC") + R"CLC(

float3 read_train_rep(__global const float* rep_positions, uint area, uint sample, uint rep, uint sample_count) {
  const uint base = ((area * sample_count + sample) * 16u + rep) * 3u;
  return (float3)(rep_positions[base], rep_positions[base + 1u], rep_positions[base + 2u]);
}

void write_train_feature(__global float* features, uint area, uint sample, uint rep, uint sample_count, float3 p) {
  const uint base = ((area * sample_count + sample) * 16u + rep) * 3u;
  features[base] = p.x;
  features[base + 1u] = p.y;
  features[base + 2u] = p.z;
}

float3 read_train_feature(__global const float* features, uint area, uint sample, uint rep, uint sample_count) {
  const uint base = ((area * sample_count + sample) * 16u + rep) * 3u;
  return (float3)(features[base], features[base + 1u], features[base + 2u]);
}

void gauss_jordan_inverse_private(
    __private float* mat,
    __private float* inv,
    uint n) {
  const uint stride = 12u;
  for (uint i = 0u; i < n; i++) {
    for (uint j = 0u; j < n; j++) {
      inv[i * stride + j] = i == j ? 1.0f : 0.0f;
    }
  }

  for (uint col = 0u; col < n; col++) {
    float max_val = fabs(mat[col * stride + col]);
    uint max_row = col;
    for (uint row = col + 1u; row < n; row++) {
      const float v = fabs(mat[row * stride + col]);
      if (v > max_val) {
        max_val = v;
        max_row = row;
      }
    }

    if (max_row != col) {
      for (uint j = 0u; j < n; j++) {
        const float tmp = mat[col * stride + j];
        mat[col * stride + j] = mat[max_row * stride + j];
        mat[max_row * stride + j] = tmp;
        const float tmp_inv = inv[col * stride + j];
        inv[col * stride + j] = inv[max_row * stride + j];
        inv[max_row * stride + j] = tmp_inv;
      }
    }

    const float pivot = mat[col * stride + col];
    if (fabs(pivot) < 1.0e-12f) {
      continue;
    }
    const float inv_pivot = 1.0f / pivot;
    for (uint j = 0u; j < n; j++) {
      mat[col * stride + j] *= inv_pivot;
      inv[col * stride + j] *= inv_pivot;
    }

    for (uint row = 0u; row < n; row++) {
      if (row == col) {
        continue;
      }
      const float factor = mat[row * stride + col];
      if (fabs(factor) < 1.0e-15f) {
        continue;
      }
      for (uint j = 0u; j < n; j++) {
        mat[row * stride + j] -= factor * mat[col * stride + j];
        inv[row * stride + j] -= factor * inv[col * stride + j];
      }
    }
  }
}

__kernel void train_areas(
    __global const float* rep_positions,
    __global float* feature_out,
    __global float* theta_out,
    __global float* scale_out,
    const uint area_count,
    const uint sample_count,
    const float rbf_radius,
    const float regularization) {
  const uint area = get_global_id(0);
  if (area >= area_count || sample_count > 12u) { return; }

  float3 bind_pts[16];
  for (uint r = 0u; r < 16u; r++) {
    bind_pts[r] = read_train_rep(rep_positions, area, 0u, r, sample_count);
    write_train_feature(feature_out, area, 0u, r, sample_count, bind_pts[r]);
  }

  for (uint s = 1u; s < sample_count; s++) {
    float3 sample_pts[16];
    for (uint r = 0u; r < 16u; r++) {
      sample_pts[r] = read_train_rep(rep_positions, area, s, r, sample_count);
    }

    float R[9];
    procrustes_rotation16(sample_pts, bind_pts, R);
    float3 sample_centroid = (float3)(0.0f, 0.0f, 0.0f);
    float3 bind_centroid = (float3)(0.0f, 0.0f, 0.0f);
    for (uint r = 0u; r < 16u; r++) {
      sample_centroid += sample_pts[r];
      bind_centroid += bind_pts[r];
    }
    sample_centroid *= 1.0f / 16.0f;
    bind_centroid *= 1.0f / 16.0f;

    for (uint r = 0u; r < 16u; r++) {
      const float3 aligned = rotate3(R, sample_pts[r] - sample_centroid) + bind_centroid;
      write_train_feature(feature_out, area, s, r, sample_count, aligned);
    }
  }

  float3 min_p = bind_pts[0];
  float3 max_p = bind_pts[0];
  for (uint r = 1u; r < 16u; r++) {
    min_p = fmin(min_p, bind_pts[r]);
    max_p = fmax(max_p, bind_pts[r]);
  }
  const float diagonal = distance(min_p, max_p);
  const float scale = diagonal > 1.0e-5f ? 1.0f / diagonal : 1.0f;
  scale_out[area] = scale;

  for (uint s = 0u; s < sample_count; s++) {
    for (uint r = 0u; r < 16u; r++) {
      const float3 p = read_train_feature(feature_out, area, s, r, sample_count) * scale;
      write_train_feature(feature_out, area, s, r, sample_count, p);
    }
  }

  float phi[12 * 12];
  for (uint i = 0u; i < sample_count; i++) {
    for (uint j = 0u; j < sample_count; j++) {
      float total = 0.0f;
      for (uint r = 0u; r < 16u; r++) {
        const float d = distance(
            read_train_feature(feature_out, area, i, r, sample_count),
            read_train_feature(feature_out, area, j, r, sample_count));
        total += 1.0f / sqrt(d * d + rbf_radius * rbf_radius);
      }
      phi[i * 12u + j] = total / 16.0f;
    }
  }

  float a[12 * 12];
  for (uint i = 0u; i < sample_count; i++) {
    for (uint j = 0u; j < sample_count; j++) {
      float sum = 0.0f;
      for (uint k = 0u; k < sample_count; k++) {
        sum += phi[k * 12u + i] * phi[k * 12u + j];
      }
      a[i * 12u + j] = sum + (i == j ? regularization : 0.0f);
    }
  }

  float a_inv[12 * 12];
  gauss_jordan_inverse_private(a, a_inv, sample_count);

  for (uint i = 0u; i < sample_count; i++) {
    for (uint j = 0u; j < sample_count; j++) {
      float sum = 0.0f;
      for (uint k = 0u; k < sample_count; k++) {
        sum += phi[i * 12u + k] * a_inv[k * 12u + j];
      }
      theta_out[(area * sample_count + i) * sample_count + j] = sum;
    }
  }
}

__kernel void evaluate_areas(
    __global const float* positions,
    __global const uint* rep_indices,
    __global const float* theta_values,
    __global const float* scale_values,
    __global const float* feature_values,
    __global float* activations,
    const uint area_count,
    const uint sample_count,
    const float rbf_radius) {
  const uint area = get_global_id(0);
  if (area >= area_count) { return; }

  float3 current_pts[16];
  float3 bind_pts[16];
  for (uint r = 0u; r < 16u; r++) {
    current_pts[r] = read_point(positions, rep_indices[area * 16u + r]);
  }

  const float scale = scale_values[area];
  const float inv_scale = fabs(scale) > 1.0e-10f ? 1.0f / scale : 1.0f;
  for (uint r = 0u; r < 16u; r++) {
    bind_pts[r] = read_feature(feature_values, area, 0u, r, sample_count) * inv_scale;
  }

  float align_R[9];
  procrustes_rotation16(current_pts, bind_pts, align_R);
  float3 current_centroid = (float3)(0.0f, 0.0f, 0.0f);
  float3 bind_centroid = (float3)(0.0f, 0.0f, 0.0f);
  for (uint r = 0u; r < 16u; r++) {
    current_centroid += current_pts[r];
    bind_centroid += bind_pts[r];
  }
  current_centroid *= 1.0f / 16.0f;
  bind_centroid *= 1.0f / 16.0f;

  float3 aligned[16];
  for (uint r = 0u; r < 16u; r++) {
    aligned[r] = (rotate3(align_R, current_pts[r] - current_centroid) + bind_centroid) * scale;
  }

  float phi[32];
  for (uint s = 0u; s < sample_count; s++) {
    float total = 0.0f;
    for (uint r = 0u; r < 16u; r++) {
      const float d = distance(aligned[r], read_feature(feature_values, area, s, r, sample_count));
      total += 1.0f / sqrt(d * d + rbf_radius * rbf_radius);
    }
    phi[s] = total / 16.0f;
  }

  float out_w[32];
  for (uint i = 0u; i < sample_count; i++) {
    float sum = 0.0f;
    for (uint j = 0u; j < sample_count; j++) {
      sum += theta_values[(area * sample_count + i) * sample_count + j] * phi[j];
    }
    out_w[i] = isfinite(sum) ? sum : 0.0f;
  }
  project_simplex_private(out_w, sample_count);
  for (uint i = 0u; i < sample_count; i++) {
    activations[area * sample_count + i] = out_w[i];
  }
}

float3 mat3_column_major_mul(__global const float* values, uint base, float3 v) {
  return (float3)(
      values[base] * v.x + values[base + 3u] * v.y + values[base + 6u] * v.z,
      values[base + 1u] * v.x + values[base + 4u] * v.y + values[base + 7u] * v.z,
      values[base + 2u] * v.x + values[base + 5u] * v.y + values[base + 8u] * v.z);
}

uint finite3(float3 v) {
  return (isfinite(v.x) != 0 && isfinite(v.y) != 0 && isfinite(v.z) != 0) ? 1u : 0u;
}

uint build_tangent_basis(float3 p1, float3 p2, float3 p3, __private float* out) {
  const float3 e1_raw = p2 - p1;
  const float3 e2_raw = p3 - p1;
  if (finite3(e1_raw) == 0u || finite3(e2_raw) == 0u) {
    return 0u;
  }
  const float3 normal = cross(e1_raw, e2_raw);
  const float area2 = length(normal);
  if (!isfinite(area2) || area2 < 1.0e-10f || finite3(normal) == 0u) {
    return 0u;
  }
  out[0] = e1_raw.x; out[1] = e1_raw.y; out[2] = e1_raw.z;
  out[3] = e2_raw.x; out[4] = e2_raw.y; out[5] = e2_raw.z;
  out[6] = normal.x; out[7] = normal.y; out[8] = normal.z;
  return 1u;
}

__kernel void compute_tangent_frames(
    __global const float* positions,
    __global const uint* frame_centers,
    __global const uint* frame_neighbor_a,
    __global const uint* frame_neighbor_b,
    __global float* current_frames,
    __global uint* current_frame_valid,
    const uint frame_count) {
  const uint frame = get_global_id(0);
  if (frame >= frame_count) { return; }
  float basis[9];
  const uint valid = build_tangent_basis(
      read_point(positions, frame_centers[frame]),
      read_point(positions, frame_neighbor_a[frame]),
      read_point(positions, frame_neighbor_b[frame]),
      basis);
  current_frame_valid[frame] = valid;
  const uint base = frame * 9u;
  for (uint i = 0u; i < 9u; i++) {
    current_frames[base + i] = valid != 0u ? basis[i] : 0.0f;
  }
}

float3 transport_delta(
    float3 delta,
    uint vertex,
    uint sample,
    __global const uint* frame_offsets,
    __global const float* sample_base_frame_inverse,
    __global const uint* sample_base_frame_valid,
    __global const float* current_frames,
    __global const uint* current_frame_valid,
    uint frame_count) {
  const float delta_len = length(delta);
  if (delta_len < 0.001f) {
    return delta;
  }

  const uint start = frame_offsets[vertex];
  const uint end = frame_offsets[vertex + 1u];
  float3 accum = (float3)(0.0f, 0.0f, 0.0f);
  uint valid_count = 0u;
  for (uint frame = start; frame < end; frame++) {
    const uint sample_frame = sample * frame_count + frame;
    if (current_frame_valid[frame] == 0u || sample_base_frame_valid[sample_frame] == 0u) {
      continue;
    }
    const float3 local_delta = mat3_column_major_mul(sample_base_frame_inverse, sample_frame * 9u, delta);
    if (finite3(local_delta) == 0u) {
      continue;
    }
    const float3 rotated = mat3_column_major_mul(current_frames, frame * 9u, local_delta);
    if (finite3(rotated) == 0u) {
      continue;
    }
    accum += rotated;
    valid_count++;
  }
  if (valid_count == 0u) {
    return delta;
  }
  if (finite3(accum) == 0u) {
    return delta;
  }
  float3 average = accum / (float)valid_count;
  const float average_len = length(average);
  if (isfinite(average_len) && average_len > 1.0e-8f) {
    average = average * (delta_len / average_len);
  }
  return average;
}

__kernel void apply_pose_trainer(
    __global const float* smoothed,
    __global const float* animated,
    __global float* output,
    __global const float* sample_deltas,
    __global const uint* membership_offsets,
    __global const uint* membership_area_ids,
    __global const float* membership_weights,
    __global const float* activations,
    __global const uint* frame_offsets,
    __global const float* sample_base_frame_inverse,
    __global const uint* sample_base_frame_valid,
    __global const float* current_frames,
    __global const uint* current_frame_valid,
    __global const float* vertex_mask,
    const uint vertex_count,
    const uint sample_count,
    const uint shape_count,
    const uint frame_count,
    const float envelope,
    const uint use_mask) {
  const uint v = get_global_id(0);
  if (v >= vertex_count) { return; }
  const uint base = v * 3u;
  const float smx = smoothed[base];
  const float smy = smoothed[base + 1u];
  const float smz = smoothed[base + 2u];
  const float anx = animated[base];
  const float any = animated[base + 1u];
  const float anz = animated[base + 2u];
  const float pdx = anx - smx;
  const float pdy = any - smy;
  const float pdz = anz - smz;

  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  float total_area_weight = 0.0f;

  const uint m0 = membership_offsets[v];
  const uint m1 = membership_offsets[v + 1u];
  for (uint i = m0; i < m1; i++) {
    const uint area = membership_area_ids[i];
    const float area_w = membership_weights[i];
    total_area_weight += area_w;

    const uint act_base = area * sample_count;
    const float bind_w = activations[act_base];
    ax += pdx * area_w * bind_w;
    ay += pdy * area_w * bind_w;
    az += pdz * area_w * bind_w;

    for (uint s = 0u; s < shape_count; s++) {
      const float activation = activations[act_base + s + 1u];
      if (activation == 0.0f) { continue; }

      const uint dbase = (s * vertex_count + v) * 3u;
      const float dx = sample_deltas[dbase];
      const float dy = sample_deltas[dbase + 1u];
      const float dz = sample_deltas[dbase + 2u];
      const float3 transported = transport_delta(
          (float3)(dx, dy, dz),
          v,
          s,
          frame_offsets,
          sample_base_frame_inverse,
          sample_base_frame_valid,
          current_frames,
          current_frame_valid,
          frame_count);
      ax += transported.x * area_w * activation;
      ay += transported.y * area_w * activation;
      az += transported.z * area_w * activation;
    }
  }

  const float uncovered = fmax(1.0f - total_area_weight, 0.0f);
  ax += pdx * uncovered;
  ay += pdy * uncovered;
  az += pdz * uncovered;

  const float cx = smx + ax;
  const float cy = smy + ay;
  const float cz = smz + az;
  const float blend = envelope * (use_mask != 0u ? vertex_mask[v] : 1.0f);
  output[base] = anx * (1.0f - blend) + cx * blend;
  output[base + 1u] = any * (1.0f - blend) + cy * blend;
  output[base + 2u] = anz * (1.0f - blend) + cz * blend;
}
)CLC";
}

}  // namespace

struct OpenClRuntime {
  enum class ProfileBucket {
    Upload,
    Relax,
    Areas,
    Apply,
    Read
  };

  struct ProfileEvent {
    cl_event event = nullptr;
    ProfileBucket bucket = ProfileBucket::Upload;
  };

  struct BucketTiming {
    double sum_ms = 0.0;
    cl_ulong first_start = std::numeric_limits<cl_ulong>::max();
    cl_ulong last_end = 0;
    uint32_t count = 0;
  };

  OpenClApi& cl;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel relax_kernel = nullptr;
  cl_kernel train_kernel = nullptr;
  cl_kernel evaluate_kernel = nullptr;
  cl_kernel frame_kernel = nullptr;
  cl_kernel apply_kernel = nullptr;

  cl_mem halfedge_twin = nullptr;
  cl_mem halfedge_next = nullptr;
  cl_mem halfedge_vertex = nullptr;
  cl_mem vertex_halfedge = nullptr;
  cl_mem membership_offsets = nullptr;
  cl_mem membership_area_ids = nullptr;
  cl_mem membership_weights = nullptr;
  cl_mem tangent_frame_offsets = nullptr;
  cl_mem tangent_frame_centers = nullptr;
  cl_mem tangent_frame_neighbor_a = nullptr;
  cl_mem tangent_frame_neighbor_b = nullptr;
  cl_mem sample_base_frame_inverse = nullptr;
  cl_mem sample_base_frame_valid = nullptr;
  cl_mem sample_deltas = nullptr;
  cl_mem rep_indices = nullptr;
  cl_mem theta_values = nullptr;
  cl_mem scale_values = nullptr;
  cl_mem feature_values = nullptr;
  cl_mem animated = nullptr;
  cl_mem ping = nullptr;
  cl_mem pong = nullptr;
  cl_mem output = nullptr;
  cl_mem mask = nullptr;
  cl_mem activations = nullptr;
  cl_mem current_frames = nullptr;
  cl_mem current_frame_valid = nullptr;

  size_t static_key = 0;
  size_t position_capacity = 0;
  size_t mask_capacity = 0;
  size_t activation_capacity = 0;
  size_t current_frame_capacity = 0;
  size_t current_frame_valid_capacity = 0;
  bool profiling_enabled = false;
  bool mask_uploaded = false;
  bool collect_profile = false;
  bool relax_static_args_dirty = true;
  size_t uploaded_mask_size = 0;
  uint64_t uploaded_mask_hash = 0;
  cl_uint relax_static_vertex_count = 0;
  std::vector<ProfileEvent> profile_events;
  std::string last_profile_timing;

  explicit OpenClRuntime(OpenClApi& api_ref) : cl(api_ref) {
    if (!cl.loaded()) {
      throw std::runtime_error(cl.status);
    }

    cl_uint platform_count = 0;
    check(cl.clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
    if (platform_count == 0) {
      throw std::runtime_error("No OpenCL platforms found");
    }
    std::vector<cl_platform_id> platforms(platform_count);
    check(cl.clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

    for (cl_platform_id platform : platforms) {
      cl_uint device_count = 0;
      cl_int err = cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
      if (err != CL_SUCCESS || device_count == 0) {
        err = cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
      }
      if (err != CL_SUCCESS || device_count == 0) {
        continue;
      }
      std::vector<cl_device_id> devices(device_count);
      check(cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr), "clGetDeviceIDs");
      device = devices[0];
      break;
    }
    if (!device) {
      throw std::runtime_error("No usable OpenCL device found");
    }

    cl_int err = CL_SUCCESS;
    context = cl.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    queue = cl.clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err == CL_SUCCESS) {
      profiling_enabled = true;
    } else {
      queue = cl.clCreateCommandQueue(context, device, 0, &err);
      check(err, "clCreateCommandQueue");
    }

    const std::string source = kernel_source();
    const char* source_ptr = source.c_str();
    const size_t source_len = source.size();
    program = cl.clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &err);
    check(err, "clCreateProgramWithSource");
    err = cl.clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t log_size = 0;
      cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::string log(log_size, '\0');
      if (log_size > 0) {
        cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
      }
      throw std::runtime_error("OpenCL kernel build failed: " + log);
    }

    relax_kernel = cl.clCreateKernel(program, "relax_neighbors", &err);
    check(err, "clCreateKernel(relax_neighbors)");
    train_kernel = cl.clCreateKernel(program, "train_areas", &err);
    check(err, "clCreateKernel(train_areas)");
    evaluate_kernel = cl.clCreateKernel(program, "evaluate_areas", &err);
    check(err, "clCreateKernel(evaluate_areas)");
    frame_kernel = cl.clCreateKernel(program, "compute_tangent_frames", &err);
    check(err, "clCreateKernel(compute_tangent_frames)");
    apply_kernel = cl.clCreateKernel(program, "apply_pose_trainer", &err);
    check(err, "clCreateKernel(apply_pose_trainer)");
  }

  ~OpenClRuntime() {
    release(halfedge_twin);
    release(halfedge_next);
    release(halfedge_vertex);
    release(vertex_halfedge);
    release(membership_offsets);
    release(membership_area_ids);
    release(membership_weights);
    release(tangent_frame_offsets);
    release(tangent_frame_centers);
    release(tangent_frame_neighbor_a);
    release(tangent_frame_neighbor_b);
    release(sample_base_frame_inverse);
    release(sample_base_frame_valid);
    release(sample_deltas);
    release(rep_indices);
    release(theta_values);
    release(scale_values);
    release(feature_values);
    release(animated);
    release(ping);
    release(pong);
    release(output);
    release(mask);
    release(activations);
    release(current_frames);
    release(current_frame_valid);
    if (relax_kernel) cl.clReleaseKernel(relax_kernel);
    if (train_kernel) cl.clReleaseKernel(train_kernel);
    if (evaluate_kernel) cl.clReleaseKernel(evaluate_kernel);
    if (frame_kernel) cl.clReleaseKernel(frame_kernel);
    if (apply_kernel) cl.clReleaseKernel(apply_kernel);
    release_profile_events();
    if (program) cl.clReleaseProgram(program);
    if (queue) cl.clReleaseCommandQueue(queue);
    if (context) cl.clReleaseContext(context);
  }

  void release(cl_mem& mem) {
    if (mem) {
      cl.clReleaseMemObject(mem);
      mem = nullptr;
    }
  }

  void release_profile_events() {
    for (ProfileEvent& item : profile_events) {
      if (item.event) {
        cl.clReleaseEvent(item.event);
        item.event = nullptr;
      }
    }
    profile_events.clear();
  }

  cl_event* profile_event_slot(ProfileBucket bucket, cl_event& event) {
    if (!collect_profile) {
      return nullptr;
    }
    event = nullptr;
    profile_events.push_back({nullptr, bucket});
    return &event;
  }

  void commit_profile_event(cl_event event) {
    if (!collect_profile) {
      return;
    }
    profile_events.back().event = event;
  }

  void begin_profile(bool enabled) {
    release_profile_events();
    last_profile_timing.clear();
    collect_profile = profiling_enabled && enabled;
  }

  void finish_profile() {
    if (!collect_profile || profile_events.empty()) {
      last_profile_timing.clear();
      release_profile_events();
      collect_profile = false;
      return;
    }

    BucketTiming upload;
    BucketTiming relax;
    BucketTiming areas;
    BucketTiming apply;
    BucketTiming read;
    cl_ulong first_start = std::numeric_limits<cl_ulong>::max();
    cl_ulong last_end = 0;

    auto add_timing = [&](BucketTiming& bucket, cl_ulong start, cl_ulong end) {
      const double elapsed_ms = static_cast<double>(end - start) * 1.0e-6;
      bucket.sum_ms += elapsed_ms;
      bucket.first_start = std::min(bucket.first_start, start);
      bucket.last_end = std::max(bucket.last_end, end);
      bucket.count++;
      first_start = std::min(first_start, start);
      last_end = std::max(last_end, end);
    };

    for (const ProfileEvent& item : profile_events) {
      if (!item.event) {
        continue;
      }
      cl_ulong start = 0;
      cl_ulong end = 0;
      const cl_int start_err = cl.clGetEventProfilingInfo(
          item.event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
      const cl_int end_err = cl.clGetEventProfilingInfo(
          item.event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
      if (start_err != CL_SUCCESS || end_err != CL_SUCCESS || end <= start) {
        continue;
      }
      switch (item.bucket) {
        case ProfileBucket::Upload: add_timing(upload, start, end); break;
        case ProfileBucket::Relax: add_timing(relax, start, end); break;
        case ProfileBucket::Areas: add_timing(areas, start, end); break;
        case ProfileBucket::Apply: add_timing(apply, start, end); break;
        case ProfileBucket::Read: add_timing(read, start, end); break;
      }
    }

    const double wall_ms = (last_end > first_start && first_start != std::numeric_limits<cl_ulong>::max())
        ? static_cast<double>(last_end - first_start) * 1.0e-6
        : 0.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << "GPU wall " << wall_ms
        << " | up " << upload.sum_ms
        << ", relax " << relax.sum_ms
        << ", areas " << areas.sum_ms
        << ", apply " << apply.sum_ms
        << ", read " << read.sum_ms << " ms";
    last_profile_timing = out.str();
    release_profile_events();
    collect_profile = false;
  }

  cl_mem ensure_buffer(cl_mem& buffer, size_t& capacity, size_t bytes) {
    if (buffer && capacity >= bytes) {
      return buffer;
    }
    release(buffer);
    cl_int err = CL_SUCCESS;
    buffer = cl.clCreateBuffer(context, CL_MEM_READ_WRITE, std::max<size_t>(bytes, 4), nullptr, &err);
    check(err, "clCreateBuffer");
    capacity = bytes;
    return buffer;
  }

  cl_mem make_static_buffer(const void* data, size_t bytes, const char* name) {
    cl_int err = CL_SUCCESS;
    cl_mem buffer = cl.clCreateBuffer(context, CL_MEM_READ_WRITE, std::max<size_t>(bytes, 4), nullptr, &err);
    check(err, name);
    if (bytes > 0) {
      check(cl.clEnqueueWriteBuffer(queue, buffer, CL_TRUE_VALUE, 0, bytes, data, 0, nullptr, nullptr), name);
    }
    return buffer;
  }

  void ensure_static_buffers(const PoseTrainerCache& cache) {
    uint64_t key = 1469598103934665603ull;
    hash_combine(key, cache.vertex_count);
    hash_combine(key, cache.areas.size());
    hash_combine(key, hash_uint_vector(cache.halfedge_twin));
    hash_combine(key, hash_uint_vector(cache.halfedge_next));
    hash_combine(key, hash_uint_vector(cache.halfedge_vertex));
    hash_combine(key, hash_uint_vector(cache.vertex_halfedge));
    hash_combine(key, hash_uint_vector(cache.membership_offsets));
    hash_combine(key, hash_uint_vector(cache.membership_area_ids));
    hash_combine(key, hash_float_vector(cache.membership_weights));
    hash_combine(key, hash_uint_vector(cache.tangent_frame_offsets));
    hash_combine(key, hash_uint_vector(cache.tangent_frame_center_indices));
    hash_combine(key, hash_uint_vector(cache.tangent_frame_neighbor_a));
    hash_combine(key, hash_uint_vector(cache.tangent_frame_neighbor_b));
    hash_combine(key, hash_float_vector(cache.sample_base_frame_inverse_flat));
    hash_combine(key, hash_uint_vector(cache.sample_base_frame_valid_flat));
    hash_combine(key, hash_float_vector(cache.sample_deltas_flat));
    hash_combine(key, hash_uint_vector(cache.rep_indices_flat));
    hash_combine(key, hash_float_vector(cache.theta_values_flat));
    hash_combine(key, hash_float_vector(cache.scale_values));
    hash_combine(key, hash_float_vector(cache.feature_values_flat));
    if (static_key == key && halfedge_twin) {
      return;
    }
    release(halfedge_twin);
    release(halfedge_next);
    release(halfedge_vertex);
    release(vertex_halfedge);
    release(membership_offsets);
    release(membership_area_ids);
    release(membership_weights);
    release(tangent_frame_offsets);
    release(tangent_frame_centers);
    release(tangent_frame_neighbor_a);
    release(tangent_frame_neighbor_b);
    release(sample_base_frame_inverse);
    release(sample_base_frame_valid);
    release(sample_deltas);
    release(rep_indices);
    release(theta_values);
    release(scale_values);
    release(feature_values);

    halfedge_twin = make_static_buffer(
        cache.halfedge_twin.data(),
        cache.halfedge_twin.size() * sizeof(uint32_t),
        "halfedge_twin");
    halfedge_next = make_static_buffer(
        cache.halfedge_next.data(),
        cache.halfedge_next.size() * sizeof(uint32_t),
        "halfedge_next");
    halfedge_vertex = make_static_buffer(
        cache.halfedge_vertex.data(),
        cache.halfedge_vertex.size() * sizeof(uint32_t),
        "halfedge_vertex");
    vertex_halfedge = make_static_buffer(
        cache.vertex_halfedge.data(),
        cache.vertex_halfedge.size() * sizeof(uint32_t),
        "vertex_halfedge");
    membership_offsets = make_static_buffer(
        cache.membership_offsets.data(),
        cache.membership_offsets.size() * sizeof(uint32_t),
        "membership_offsets");
    membership_area_ids = make_static_buffer(
        cache.membership_area_ids.data(),
        cache.membership_area_ids.size() * sizeof(uint32_t),
        "membership_area_ids");
    membership_weights = make_static_buffer(
        cache.membership_weights.data(),
        cache.membership_weights.size() * sizeof(float),
        "membership_weights");
    tangent_frame_offsets = make_static_buffer(
        cache.tangent_frame_offsets.data(),
        cache.tangent_frame_offsets.size() * sizeof(uint32_t),
        "tangent_frame_offsets");
    tangent_frame_centers = make_static_buffer(
        cache.tangent_frame_center_indices.data(),
        cache.tangent_frame_center_indices.size() * sizeof(uint32_t),
        "tangent_frame_centers");
    tangent_frame_neighbor_a = make_static_buffer(
        cache.tangent_frame_neighbor_a.data(),
        cache.tangent_frame_neighbor_a.size() * sizeof(uint32_t),
        "tangent_frame_neighbor_a");
    tangent_frame_neighbor_b = make_static_buffer(
        cache.tangent_frame_neighbor_b.data(),
        cache.tangent_frame_neighbor_b.size() * sizeof(uint32_t),
        "tangent_frame_neighbor_b");
    sample_base_frame_inverse = make_static_buffer(
        cache.sample_base_frame_inverse_flat.data(),
        cache.sample_base_frame_inverse_flat.size() * sizeof(float),
        "sample_base_frame_inverse");
    sample_base_frame_valid = make_static_buffer(
        cache.sample_base_frame_valid_flat.data(),
        cache.sample_base_frame_valid_flat.size() * sizeof(uint32_t),
        "sample_base_frame_valid");
    sample_deltas = make_static_buffer(
        cache.sample_deltas_flat.data(),
        cache.sample_deltas_flat.size() * sizeof(float),
        "sample_deltas");
    rep_indices = make_static_buffer(
        cache.rep_indices_flat.data(),
        cache.rep_indices_flat.size() * sizeof(uint32_t),
        "rep_indices");
    theta_values = make_static_buffer(
        cache.theta_values_flat.data(),
        cache.theta_values_flat.size() * sizeof(float),
        "theta_values");
    scale_values = make_static_buffer(
        cache.scale_values.data(),
        cache.scale_values.size() * sizeof(float),
        "scale_values");
    feature_values = make_static_buffer(
        cache.feature_values_flat.data(),
        cache.feature_values_flat.size() * sizeof(float),
        "feature_values");
    static_key = key;
    relax_static_args_dirty = true;
  }

  void ensure_relax_static_args(cl_uint vertex_count) {
    if (!relax_static_args_dirty && relax_static_vertex_count == vertex_count) {
      return;
    }
    cl_uint arg = 2;
    check(cl.clSetKernelArg(relax_kernel, arg++, sizeof(cl_mem), &halfedge_twin), "relax arg halfedge_twin");
    check(cl.clSetKernelArg(relax_kernel, arg++, sizeof(cl_mem), &halfedge_next), "relax arg halfedge_next");
    check(cl.clSetKernelArg(relax_kernel, arg++, sizeof(cl_mem), &halfedge_vertex), "relax arg halfedge_vertex");
    check(cl.clSetKernelArg(relax_kernel, arg++, sizeof(cl_mem), &vertex_halfedge), "relax arg vertex_halfedge");
    check(cl.clSetKernelArg(relax_kernel, arg++, sizeof(cl_uint), &vertex_count), "relax arg vertex_count");
    relax_static_vertex_count = vertex_count;
    relax_static_args_dirty = false;
  }

  void write_buffer(cl_mem buffer, const void* data, size_t bytes, const char* name) {
    if (bytes == 0) {
      return;
    }
    cl_event event = nullptr;
    check(cl.clEnqueueWriteBuffer(
        queue,
        buffer,
        CL_FALSE_VALUE,
        0,
        bytes,
        data,
        0,
        nullptr,
        profile_event_slot(ProfileBucket::Upload, event)),
        name);
    commit_profile_event(event);
  }

  void read_buffer(cl_mem buffer, void* data, size_t bytes, const char* name) {
    if (bytes == 0) {
      return;
    }
    cl_event event = nullptr;
    check(cl.clEnqueueReadBuffer(
        queue,
        buffer,
        CL_TRUE_VALUE,
        0,
        bytes,
        data,
        0,
        nullptr,
        profile_event_slot(ProfileBucket::Read, event)),
        name);
    commit_profile_event(event);
  }

  cl_mem dispatch_relax(cl_mem input, cl_uint vertex_count, int iterations) {
    if (iterations <= 0) {
      return input;
    }
    cl_mem result = input;
    const size_t global = ((static_cast<size_t>(vertex_count) + 255) / 256) * 256;
    const size_t local = 256;
    ensure_relax_static_args(vertex_count);
    for (int i = 0; i < iterations; ++i) {
      const cl_mem read_buf = (i == 0) ? input : result;
      const cl_mem write_buf = (i % 2 == 0) ? ping : pong;
      cl_uint arg = 0;
      check(cl.clSetKernelArg(relax_kernel, arg++, sizeof(cl_mem), &read_buf), "relax arg input");
      check(cl.clSetKernelArg(relax_kernel, arg++, sizeof(cl_mem), &write_buf), "relax arg output");
      cl_event event = nullptr;
      check(cl.clEnqueueNDRangeKernel(
          queue,
          relax_kernel,
          1,
          nullptr,
          &global,
          &local,
          0,
          nullptr,
          profile_event_slot(ProfileBucket::Relax, event)),
          "relax kernel");
      commit_profile_event(event);
      result = write_buf;
    }
    return result;
  }

  std::vector<Vec3> relax_points(
      const PoseTrainerCache& cache,
      const std::vector<Vec3>& points,
      int iterations) {
    if (points.size() != cache.vertex_count) {
      throw std::invalid_argument("OpenCL relax point count does not match cache");
    }
    ensure_static_buffers(cache);
    const size_t position_bytes = cache.vertex_count * 3 * sizeof(float);
    ensure_buffer(animated, position_capacity, position_bytes);
    ensure_buffer(ping, position_capacity, position_bytes);
    ensure_buffer(pong, position_capacity, position_bytes);

    write_buffer(animated, points.data(), position_bytes, "write train relax points");
    cl_mem result_buffer = dispatch_relax(animated, cache.vertex_count, iterations);
    std::vector<Vec3> result(cache.vertex_count);
    read_buffer(result_buffer, result.data(), position_bytes, "read train relaxed points");
    return result;
  }

  void train_area_models(
      PoseTrainerCache& cache,
      const std::vector<float>& rep_positions,
      int sample_count) {
    constexpr int kMaxOpenClTrainSamples = 12;
    if (sample_count <= 0 || sample_count > kMaxOpenClTrainSamples) {
      throw std::invalid_argument("OpenCL training supports 1 to 12 bind/sample entries");
    }

    const cl_uint area_count = static_cast<cl_uint>(cache.areas.size());
    const cl_uint samples = static_cast<cl_uint>(sample_count);
    if (area_count == 0) {
      return;
    }
    const size_t expected_rep_floats =
        static_cast<size_t>(area_count) * sample_count * 16 * 3;
    if (rep_positions.size() != expected_rep_floats) {
      throw std::invalid_argument("OpenCL training rep position buffer has an unexpected size");
    }

    const size_t rep_bytes = rep_positions.size() * sizeof(float);
    const size_t feature_bytes =
        static_cast<size_t>(area_count) * sample_count * 16 * 3 * sizeof(float);
    const size_t theta_bytes =
        static_cast<size_t>(area_count) * sample_count * sample_count * sizeof(float);
    const size_t scale_bytes = static_cast<size_t>(area_count) * sizeof(float);

    cl_mem rep_buffer = nullptr;
    cl_mem feature_buffer = nullptr;
    cl_mem theta_buffer = nullptr;
    cl_mem scale_buffer = nullptr;
    auto release_local = [&]() {
      release(rep_buffer);
      release(feature_buffer);
      release(theta_buffer);
      release(scale_buffer);
    };

    try {
      rep_buffer = make_static_buffer(rep_positions.data(), rep_bytes, "train rep positions");
      size_t feature_capacity = 0;
      size_t theta_capacity = 0;
      size_t scale_capacity = 0;
      ensure_buffer(feature_buffer, feature_capacity, feature_bytes);
      ensure_buffer(theta_buffer, theta_capacity, theta_bytes);
      ensure_buffer(scale_buffer, scale_capacity, scale_bytes);

      const float radius = cache.settings.rbf_radius;
      const float regularization = cache.settings.regularization;
      cl_uint arg = 0;
      check(cl.clSetKernelArg(train_kernel, arg++, sizeof(cl_mem), &rep_buffer), "train arg reps");
      check(cl.clSetKernelArg(train_kernel, arg++, sizeof(cl_mem), &feature_buffer), "train arg features");
      check(cl.clSetKernelArg(train_kernel, arg++, sizeof(cl_mem), &theta_buffer), "train arg theta");
      check(cl.clSetKernelArg(train_kernel, arg++, sizeof(cl_mem), &scale_buffer), "train arg scale");
      check(cl.clSetKernelArg(train_kernel, arg++, sizeof(cl_uint), &area_count), "train arg area_count");
      check(cl.clSetKernelArg(train_kernel, arg++, sizeof(cl_uint), &samples), "train arg sample_count");
      check(cl.clSetKernelArg(train_kernel, arg++, sizeof(float), &radius), "train arg radius");
      check(cl.clSetKernelArg(train_kernel, arg++, sizeof(float), &regularization), "train arg regularization");

      const size_t global = std::max<size_t>(area_count, 1);
      check(cl.clEnqueueNDRangeKernel(
          queue,
          train_kernel,
          1,
          nullptr,
          &global,
          nullptr,
          0,
          nullptr,
          nullptr),
          "train areas kernel");

      std::vector<float> features(static_cast<size_t>(area_count) * sample_count * 16 * 3);
      std::vector<float> theta(static_cast<size_t>(area_count) * sample_count * sample_count);
      std::vector<float> scales(area_count, 1.0f);
      check(cl.clEnqueueReadBuffer(
          queue, feature_buffer, CL_TRUE_VALUE, 0, feature_bytes, features.data(), 0, nullptr, nullptr),
          "read train features");
      check(cl.clEnqueueReadBuffer(
          queue, theta_buffer, CL_TRUE_VALUE, 0, theta_bytes, theta.data(), 0, nullptr, nullptr),
          "read train theta");
      check(cl.clEnqueueReadBuffer(
          queue, scale_buffer, CL_TRUE_VALUE, 0, scale_bytes, scales.data(), 0, nullptr, nullptr),
          "read train scale");

      for (AreaModel& area : cache.areas) {
        const size_t area_id = area.area_id;
        area.scale = scales[area_id];
        area.features.assign(sample_count * kRepresentativeVertexCount, {});
        for (int s = 0; s < sample_count; ++s) {
          for (int r = 0; r < kRepresentativeVertexCount; ++r) {
            const size_t src = ((area_id * sample_count + s) * kRepresentativeVertexCount + r) * 3;
            area.features[s * kRepresentativeVertexCount + r] = {
                features[src],
                features[src + 1],
                features[src + 2]};
          }
        }
        area.theta.assign(sample_count * sample_count, 0.0f);
        for (int row = 0; row < sample_count; ++row) {
          for (int col = 0; col < sample_count; ++col) {
            area.theta[row * sample_count + col] =
                theta[(area_id * sample_count + row) * sample_count + col];
          }
        }
      }
      release_local();
    } catch (...) {
      release_local();
      throw;
    }
  }

  void dispatch_evaluate_areas(cl_mem relaxed, const PoseTrainerCache& cache) {
    const cl_uint area_count = static_cast<cl_uint>(cache.areas.size());
    const cl_uint sample_count = static_cast<cl_uint>(cache.sample_deltas.size() + 1);
    const float radius = cache.settings.rbf_radius;
    const size_t activation_bytes = static_cast<size_t>(area_count) * sample_count * sizeof(float);
    ensure_buffer(activations, activation_capacity, activation_bytes);

    const size_t global = std::max<size_t>(area_count, 1);
    cl_uint arg = 0;
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(cl_mem), &relaxed), "eval arg positions");
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(cl_mem), &rep_indices), "eval arg reps");
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(cl_mem), &theta_values), "eval arg theta");
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(cl_mem), &scale_values), "eval arg scale");
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(cl_mem), &feature_values), "eval arg features");
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(cl_mem), &activations), "eval arg activations");
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(cl_uint), &area_count), "eval arg area count");
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(cl_uint), &sample_count), "eval arg sample count");
    check(cl.clSetKernelArg(evaluate_kernel, arg++, sizeof(float), &radius), "eval arg radius");
    cl_event event = nullptr;
    check(cl.clEnqueueNDRangeKernel(
        queue,
        evaluate_kernel,
        1,
        nullptr,
        &global,
        nullptr,
        0,
        nullptr,
        profile_event_slot(ProfileBucket::Areas, event)),
        "evaluate areas kernel");
    commit_profile_event(event);
  }

  void dispatch_compute_tangent_frames(cl_mem relaxed, const PoseTrainerCache& cache) {
    const cl_uint frame_count = static_cast<cl_uint>(cache.tangent_frame_center_indices.size());
    const size_t frame_bytes = static_cast<size_t>(frame_count) * 9 * sizeof(float);
    const size_t valid_bytes = static_cast<size_t>(frame_count) * sizeof(uint32_t);
    ensure_buffer(current_frames, current_frame_capacity, frame_bytes);
    ensure_buffer(current_frame_valid, current_frame_valid_capacity, valid_bytes);
    if (frame_count == 0) {
      return;
    }

    const size_t global = ((static_cast<size_t>(frame_count) + 255) / 256) * 256;
    const size_t local = 256;
    cl_uint arg = 0;
    check(cl.clSetKernelArg(frame_kernel, arg++, sizeof(cl_mem), &relaxed), "frame arg positions");
    check(cl.clSetKernelArg(frame_kernel, arg++, sizeof(cl_mem), &tangent_frame_centers), "frame arg centers");
    check(cl.clSetKernelArg(frame_kernel, arg++, sizeof(cl_mem), &tangent_frame_neighbor_a), "frame arg neighbor a");
    check(cl.clSetKernelArg(frame_kernel, arg++, sizeof(cl_mem), &tangent_frame_neighbor_b), "frame arg neighbor b");
    check(cl.clSetKernelArg(frame_kernel, arg++, sizeof(cl_mem), &current_frames), "frame arg current frames");
    check(cl.clSetKernelArg(frame_kernel, arg++, sizeof(cl_mem), &current_frame_valid), "frame arg valid");
    check(cl.clSetKernelArg(frame_kernel, arg++, sizeof(cl_uint), &frame_count), "frame arg count");
    cl_event event = nullptr;
    check(cl.clEnqueueNDRangeKernel(
        queue,
        frame_kernel,
        1,
        nullptr,
        &global,
        &local,
        0,
        nullptr,
        profile_event_slot(ProfileBucket::Apply, event)),
        "compute tangent frames kernel");
    commit_profile_event(event);
  }

  void dispatch_apply(
      cl_mem relaxed,
      const PoseTrainerCache& cache,
      const std::vector<float>& vertex_mask,
      float envelope) {
    if (!vertex_mask.empty()) {
      ensure_buffer(mask, mask_capacity, vertex_mask.size() * sizeof(float));
      const uint64_t mask_hash = hash_float_vector(vertex_mask);
      if (!mask_uploaded || uploaded_mask_size != vertex_mask.size() || uploaded_mask_hash != mask_hash) {
        write_buffer(mask, vertex_mask.data(), vertex_mask.size() * sizeof(float), "write mask");
        mask_uploaded = true;
        uploaded_mask_size = vertex_mask.size();
        uploaded_mask_hash = mask_hash;
      }
    }

    cl_uint vertex_count = cache.vertex_count;
    cl_uint sample_count = static_cast<cl_uint>(cache.sample_deltas.size() + 1);
    cl_uint shape_count = static_cast<cl_uint>(cache.sample_deltas.size());
    cl_uint frame_count = static_cast<cl_uint>(cache.tangent_frame_center_indices.size());
    cl_uint use_mask = vertex_mask.empty() ? 0u : 1u;
    const size_t global = ((static_cast<size_t>(vertex_count) + 255) / 256) * 256;
    const size_t local = 256;
    cl_mem mask_arg = mask;

    cl_uint arg = 0;
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &relaxed), "apply arg smoothed");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &animated), "apply arg animated");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &output), "apply arg output");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &sample_deltas), "apply arg deltas");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &membership_offsets), "apply arg offsets");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &membership_area_ids), "apply arg area ids");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &membership_weights), "apply arg weights");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &activations), "apply arg activations");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &tangent_frame_offsets), "apply arg frame offsets");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &sample_base_frame_inverse), "apply arg base frame inverse");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &sample_base_frame_valid), "apply arg base frame valid");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &current_frames), "apply arg current frames");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &current_frame_valid), "apply arg current frame valid");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_mem), &mask_arg), "apply arg mask");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_uint), &vertex_count), "apply arg vertex_count");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_uint), &sample_count), "apply arg sample_count");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_uint), &shape_count), "apply arg shape_count");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_uint), &frame_count), "apply arg frame_count");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(float), &envelope), "apply arg envelope");
    check(cl.clSetKernelArg(apply_kernel, arg++, sizeof(cl_uint), &use_mask), "apply arg use_mask");
    cl_event event = nullptr;
    check(cl.clEnqueueNDRangeKernel(
        queue,
        apply_kernel,
        1,
        nullptr,
        &global,
        &local,
        0,
        nullptr,
        profile_event_slot(ProfileBucket::Apply, event)),
        "apply kernel");
    commit_profile_event(event);
  }

  std::vector<Vec3> evaluate(
      const PoseTrainerCache& cache,
      const std::vector<Vec3>& animated_cpu,
      const std::vector<float>& vertex_mask,
      float envelope,
      int solve_iterations,
      bool profile_timing) {
    ensure_static_buffers(cache);
    begin_profile(profile_timing);

    const size_t position_bytes = cache.vertex_count * 3 * sizeof(float);
    ensure_buffer(animated, position_capacity, position_bytes);
    ensure_buffer(ping, position_capacity, position_bytes);
    ensure_buffer(pong, position_capacity, position_bytes);
    ensure_buffer(output, position_capacity, position_bytes);

    write_buffer(animated, animated_cpu.data(), position_bytes, "write animated");

    cl_mem current_buf = animated;
    cl_mem relaxed_buf = animated;

    for (int iter = 0; iter < solve_iterations; ++iter) {
      relaxed_buf = dispatch_relax(current_buf, cache.vertex_count, cache.settings.relax_iterations);
      dispatch_evaluate_areas(relaxed_buf, cache);
      dispatch_compute_tangent_frames(relaxed_buf, cache);
      dispatch_apply(relaxed_buf, cache, vertex_mask, envelope);
      current_buf = output;
    }

    std::vector<Vec3> result(cache.vertex_count);
    read_buffer(output, result.data(), position_bytes, "read output");
    finish_profile();
    return result;
  }
};

std::shared_ptr<OpenClRuntime> shared_opencl_runtime() {
  static std::shared_ptr<OpenClRuntime> runtime;
  if (!runtime) {
    runtime = std::make_shared<OpenClRuntime>(api());
  }
  return runtime;
}

std::vector<Vec3> evaluate_opencl(
    const PoseTrainerCache& cache,
    const std::vector<Vec3>& animated,
    const std::vector<float>& vertex_mask,
    float envelope,
    int solve_iterations,
    bool profile_timing) {
  if (cache.vertex_count == 0) {
    return {};
  }
  if (!cache.opencl_runtime) {
    cache.opencl_runtime = shared_opencl_runtime();
  }
  std::vector<Vec3> out = cache.opencl_runtime->evaluate(
      cache, animated, vertex_mask, envelope, solve_iterations, profile_timing);
  cache.last_opencl_timing = cache.opencl_runtime->last_profile_timing;
  return out;
}

std::vector<Vec3> relax_opencl(
    const PoseTrainerCache& cache,
    const std::vector<Vec3>& points,
    int iterations) {
  if (cache.vertex_count == 0) {
    return {};
  }
  if (!cache.opencl_runtime) {
    cache.opencl_runtime = shared_opencl_runtime();
  }
  return cache.opencl_runtime->relax_points(cache, points, iterations);
}

void train_area_models_opencl(
    PoseTrainerCache& cache,
    const std::vector<float>& rep_positions,
    int sample_count) {
  if (cache.vertex_count == 0 || cache.areas.empty()) {
    return;
  }
  if (!cache.opencl_runtime) {
    cache.opencl_runtime = shared_opencl_runtime();
  }
  cache.opencl_runtime->train_area_models(cache, rep_positions, sample_count);
}

bool opencl_runtime_available() {
  try {
    if (!api().loaded()) {
      return false;
    }
    shared_opencl_runtime();
    api().status = "OpenCL runtime initialized";
    return true;
  } catch (const std::exception& exc) {
    api().status = exc.what();
    return false;
  }
}

std::string opencl_runtime_status() {
  OpenClApi& opencl = api();
  if (!opencl.loaded()) {
    return opencl.status;
  }
  return opencl.status;
}

}  // namespace pose_trainer
