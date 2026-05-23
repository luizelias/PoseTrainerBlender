# Blender Pose Trainer - Codex Project Guide

## Project Purpose

This project implements a Blender-friendly Pose Trainer inspired by the Mush3D-3
Pose Trainer reference implementation. The goal is to build an example-trained
pose-space Delta Mush deformer for Blender users without requiring a custom
Blender build.

Do not implement this as a native Blender C/C++ modifier. Native modifiers require patching and
rebuilding Blender, which is not acceptable for normal user distribution.

Preferred product shape:

- A regular Blender Python add-on.
- A compiled C++ Python extension for all training/evaluation deformation math.
- A generated output mesh object that mirrors the source object after Pose
  Trainer deformation.
- Optional bake/export tools for delivery.

## Current Project State

This handoff was last updated after replacing the generic neighbor-CSR relax
path with Mush3D-style half-edge relax traversal in both CPU and OpenCL paths.
The OpenCL profiler was then corrected to use `CL_PROFILING_COMMAND_START`
instead of the queued timestamp, so `relax` no longer includes accumulated queue
wait time.

Implemented in this repository:

- Project/build scaffold:
  - `pyproject.toml`
  - `CMakeLists.txt`
  - `.gitignore`
  - `README.md`
- Blender add-on package:
  - `src/pose_trainer_blender`
  - UI tab: `View3D > Sidebar > Pose Trainer`
  - Properties for source, bind, samples, output, areas, mask, envelope, relax,
    solve, RBF radius, and regularization.
  - Operators for adding samples/areas, creating output, training, evaluating,
    and toggling live update.
  - Operator/UI button for extracting Pose Trainer area masks from the source
    object's active UV shells. It creates `PT_UVShell_###` vertex groups and
    adds them as deformation areas, using Mush-style per-vertex normalization
    across adjacent face shells.
  - Runtime glue reads evaluated source/bind/sample meshes and writes an output
    mesh object.
  - Live update is registered on both `depsgraph_update_post` and
    `frame_change_post`, so timeline scrubbing/playback should refresh the
    generated output mesh after training.
  - Envelope is a Blender property in the UI. It is passed to the C++ core as
    `mix(evaluated_source, pose_trainer_result, envelope * vertex_mask)`, so
    `0.0` is source passthrough and `1.0` is full deformation. Changing the
    property after training triggers a fresh output evaluation.
  - `Area Relax` is a Blender property passed into the C++ core as
    `PoseTrainerSettings.area_relax_iterations`. It blurs deformation-area
    weights on the mesh topology before representative vertex sampling and RBF
    training, matching the Mush-style mask relax workflow.
  - `Profile Timing` shows the last training/evaluation timing breakdown in
    separate readable UI lines: total/backend, Blender/core glue timing, and
    OpenCL GPU event buckets when the OpenCL backend is used. The GPU line is
    formatted as `GPU wall ... | up ..., relax ..., areas ..., apply ..., read
    ... ms`; bucket values are summed command execution time, not queued wait.
  - The panel includes a `Copy Timing` button that copies the latest train/eval
    timing breakdown to the system clipboard.
  - Runtime evaluation reads only evaluated positions by default. Face lists are
    read only when creating/rebuilding the output mesh topology.
  - The C++ core packs nonzero active vertices/weights per deformation area at
    training time and precomputes total area coverage. Runtime no longer scans
    every vertex for every area just to skip zero weights.
  - The C++ extension now has an optional OpenCL backend. It dynamically loads
    `OpenCL.dll`/`libOpenCL`, so no OpenCL headers or link libraries are
    required at build time.
  - The C++ core builds Mush3D-compatible half-edge topology tables:
    `halfedge_twin`, `halfedge_next`, `halfedge_vertex`, and
    `vertex_halfedge`. Boundary vertices intentionally prefer a boundary
    outgoing half-edge, matching `REFERENCE/wasm/mesh/HalfEdgeBuilder.cpp`.
  - Mesh relaxation now follows the reference `pose-trainer-relax.wgsl`
    traversal, including the boundary walk via previous half-edge -> twin.
    CPU training relax, CPU fallback evaluation relax, and OpenCL runtime relax
    use the same half-edge semantics. Do not replace this with a generic
    adjacency average or fused shortcut, because matching Mush3D deformation is
    higher priority than relax-specific micro-optimizations.
  - Area mask relaxation was also moved to the same half-edge one-ring averaging
    behavior used by Mush3D dense area relax, instead of self-inclusive neighbor
    averaging.
  - OpenCL currently accelerates runtime mesh relaxation, per-area RBF
    activation, per-area Procrustes rotation, and final per-vertex delta
    application. Runtime reads back only the final output positions for Blender
    mesh write-back. The OpenCL queue now avoids hard `clFinish` calls between
    relax/evaluate/apply kernels, uploads the animated positions once per
    evaluation, reuses the mask buffer unless the mask values change, and caches
    static relax kernel arguments so each half-edge relax iteration only resets
    the ping-pong input/output buffers.
  - Blender UI has a `Backend` selector: Auto, CPU, OpenCL. Auto tries OpenCL
    and falls back to CPU; OpenCL requires the backend and reports an error if
    unavailable.
- C++ Python extension:
  - `src/pose_trainer_core/cpp`
  - Uses Eigen explicitly for Procrustes/SVD and RBF linear solves.
  - Exposes `PoseTrainerSettings`, `PoseTrainerCache.evaluate`, `train`, and
    `project_simplex` through pybind11.
  - Compiles and passes tests with the Codex bundled Python 3.12.
- Python core fallback:
  - `src/pose_trainer_core/_fallback_core.py`
  - Import-only/error stub. It must not deform. It raises a clear error when
    the compiled extension is missing.
- Tests:
- `tests/test_core_contract.py`
- Verified against the compiled extension with `4 passed`.
- Python source syntax check:
  - `python -m compileall -q src\pose_trainer_blender`
- Packaging helper:
  - `tools/package_addon.py`
  - Builds `dist/pose_trainer_blender_addon.zip`.
  - Packages a nested `pose_trainer_core` package inside the add-on when a
    Blender-compatible wheel has been unpacked to `build_support/wheel_cp313`.

Installed/verified locally:

- Blender is installed at:
  `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`
- The Blender MCP server is installed and registered with Codex as `blender`.
- The Blender MCP bridge add-on is installed/enabled in Blender 5.1.
- The Pose Trainer add-on zip has been installed/enabled in Blender 5.1.
- The Pose Trainer UI is visible in Blender.
- `_pose_trainer_core.cp313-win_amd64.pyd` was built for Blender 5.1's Python
  3.13 ABI and packaged into `dist/pose_trainer_blender_addon.zip`.
- Live MCP verification showed:
  - `pose_trainer_blender` enabled: true
  - `scene.pose_trainer` registered: true
  - `bpy.ops.pose_trainer.train` registered: true
  - `bpy.ops.pose_trainer.create_output` registered: true
  - `pose_trainer_blender.pose_trainer_core` loaded with `USING_FALLBACK=False`.
- A live Blender smoke test created:
  - `PT_Source`
  - `PT_Bind`
  - `PT_Sample`
  - `PT_Source_PoseTrainer`
  Training/evaluation succeeded with status:
  `Trained 1 sample(s), 1 area(s) with Eigen C++ core`.
- UV shell area extraction has source-level tests only so far. Repackage and
  reload the add-on in Blender after any source change.
- A Blender 5.1 background smoke test for
  `bpy.ops.pose_trainer.extract_areas_from_uv_shells()` created two
  `PT_UVShell_###` groups from a seam-split UV mesh. Seam vertices were weighted
  `0.5 / 0.5`, confirming Mush-style adjacent-face shell normalization.
- A Blender 5.1 background smoke test trained a simple shape-key animated mesh,
  enabled live update, changed frames, and confirmed the output mesh vertices
  changed. Both live handlers were registered.
- A Blender 5.1 background smoke test confirmed Envelope `0.0` exactly matches
  the source mesh and Envelope `1.0` differs with the full deformation.
- The local C++ test suite includes a contract test confirming that
  `area_relax_iterations` changes the trained/evaluated result. The Blender 5.1
  Python 3.13 binary was rebuilt and verified to expose
  `PoseTrainerSettings.area_relax_iterations`.
- Profiling diagnostics are Python-side only and do not change C++ math. Use
  the `Profile Timing` checkbox in the Pose Trainer panel to identify whether
  the bottleneck is Blender read/write/mask glue or C++ evaluation.
- A profile from `SAM.blend` showed roughly `Eval 203 ms | C++ 197 ms, write
  4 ms`, so the current bottleneck is the C++ evaluation path, not Blender mesh
  write-back. The first active-vertex packing optimization was built for both
  local Python 3.12 and Blender 5.1/Python 3.13, and packaged into
  `dist/pose_trainer_blender_addon.zip`. Blender was open, so the installed
  `.pyd` could not yet be replaced in-place.
- Local Python 3.12 and Blender 5.1/Python 3.13 builds verified OpenCL is
  available on this machine. Tiny exact-sample CPU-vs-OpenCL output matched
  exactly; a 16-vertex non-flat grid matched within about `1.6e-6`. After the
  queue cleanup, local tests passed and a tiny OpenCL smoke run reported timing
  like `GPU up ..., relax ..., areas ..., apply ..., read ... ms`.
- The Blender 5.1/Python 3.13 wheel was rebuilt after the OpenCL queue cleanup,
  `dist/pose_trainer_blender_addon.zip` was regenerated, the installed Blender
  5.1 add-on folder was updated while Blender was closed, and a background
  Blender smoke test reported OpenCL backend plus GPU timing successfully.
- The Blender 5.1/Python 3.13 wheel was rebuilt again after the half-edge relax
  port, the installed add-on was updated, and a background Blender smoke test
  confirmed CPU/OpenCL output matched exactly on the test mesh with OpenCL
  initialized.
- The Blender 5.1/Python 3.13 wheel was rebuilt after the OpenCL profiler fix
  and relax static-argument cleanup. Background Blender smoke test reported
  `GPU wall ... | ...` timing and CPU/OpenCL max diff `0.0` on the test mesh.
- Reinstalling the full zip while another Blender process had the core `.pyd`
  loaded hit a Windows file lock. The installed add-on's Python files were
  updated in place under Blender 5.1's user add-ons folder. The `.pyd` may still
  be locked while Blender is running; rerun the normal zip install after closing
  Blender so the new C++ core binary is copied into the installed add-on.

Important current limitations:

- The current implementation is a minimal working deformation path, not a
  production-ready Pose Trainer.
- Area weights are still passed as dense per-area arrays; optimized CSR packing
  and cache serialization are not implemented yet.
- Cache invalidation/topology fingerprinting is not complete.
- Changing `Area Relax`, mesh relax iterations, RBF radius, or regularization
  marks the cache stale and requires retraining.
- UV shell extraction currently creates Blender vertex groups from the source
  mesh's active UV map; it has not yet been stress-tested on complex mirrored,
  stacked, or UDIM layouts in Blender.
- The C++ math path has only tiny contract tests and a tiny Blender smoke test;
  it still needs broader validation against the reference algorithm and more
  realistic meshes.
- Live depsgraph update exists but needs more stress testing with real animated
  rigs/modifier stacks.

Recommended next step:

1. Test the half-edge OpenCL relax timing on a real rig and compare deformation
   visually against the previous build/Mush3D expectations.
2. Test the new `GPU up/relax/areas/apply/read` timing buckets on a real rig and
   use the largest bucket to guide the next performance pass.
3. Improve topology/cache validation and error reporting.
4. Implement training cache save/load and invalidation fingerprints.
5. Add more C++ tests for Procrustes, RBF activation, simplex projection, and
   bind/sample weighting behavior.
6. Test live depsgraph updates with actual Blender animation/modifier stacks.
7. Then add bake/export tooling and performance tuning/parallelism.

## Performance Boundary

Aim for production speed from the start. The live deformation path and training
math must run in the compiled C++ Python extension, not in Python loops and not
through Python numerical libraries such as NumPy/SciPy.

Python may:

- Register Blender UI/operators/properties.
- Read evaluated Blender meshes and vertex groups.
- Marshal bulk arrays into the extension.
- Write the extension's output positions back to the generated output mesh.
- Run tiny diagnostic checks or tests when explicitly marked as non-production.

Python must not:

- Relax meshes, solve RBFs, run Procrustes, project activations, or apply
  per-vertex corrective deltas in the add-on runtime.
- Silently fall back to Python deformation when the compiled extension is
  missing. The add-on should report a clear build/install error instead.
- Use Python math libraries as the production deformer backend.

C++ should own the data-oriented hot path. Eigen is the required C++ linear
algebra library for matrix, SVD, Procrustes, RBF solve, and vectorized linear
algebra code. Keep the C++ API array-based and Blender-independent.

## Reference Repository

Use the REFERENCE folder repo as the algorithm and architecture reference. Treat
it as read-only.

Most important Mush3D-3 reference files:

- `PoseTrainer.md`: ground-truth algorithm pseudocode.
- `ui/src/scene/PoseTrainer.ts`: runtime data model.
- `ui/src/bridge/handlers/poseTrainerHandler.ts`: bind/sample/training flow.
- `ui/src/engine/NodeGraph.ts`: runtime evaluation, especially
  `evaluatePoseTrainer`.
- `ui/src/renderer/PoseTrainerDispatch.ts`: relaxation and apply dispatch shape.
- `ui/src/renderer/AutoBlendGPU.ts`: RBF training/evaluation model.
- `ui/src/shaders/pose-trainer-relax.wgsl`: Laplacian relaxation.
- `ui/src/shaders/pose-trainer-apply.wgsl`: final geometry application.
- `ui/src/shaders/auto-blend-train.wgsl`: per-area RBF training.
- `ui/src/shaders/auto-blend-evaluate.wgsl`: per-frame RBF solve.

## Core Algorithm

The deformer is an example-trained pose-space Delta Mush:

1. The user chooses a source object, a bind mesh object, and corrective sample
   mesh objects.
2. Bind and sample meshes must share topology, vertex count, and vertex order
   with the evaluated source mesh.
3. Training relaxes bind and samples with Laplacian smoothing.
4. Each corrective stores:
   `sample_delta = sample_original - sample_relaxed`.
5. Deformation areas are represented by Blender vertex groups.
6. Each area samples 16 representative vertices.
7. Each area trains an RBF over relaxed bind/sample features.
8. Runtime evaluation relaxes the current animated source mesh, solves per-area
   activations, rotates corrective deltas into the current pose with Procrustes,
   preserves current animated detail through the bind activation, and writes the
   final positions to an output mesh.

Runtime evaluation contract:

```text
animated = evaluated source positions
current = animated

for solve iteration:
  relaxed = laplacian_relax(current)
  pose_delta = animated - relaxed

  for each deformation area:
    gather 16 representative relaxed vertices
    align current feature to bind feature
    evaluate RBF weights for bind + samples
    project weights to probability simplex
    compute sample-to-current Procrustes rotations

  for each vertex:
    accum = uncovered_weight * pose_delta

    for each area membership:
      accum += area_weight * bind_weight * pose_delta
      accum += area_weight * sample_weight * rotated_sample_delta

    corrected = relaxed + accum
    output = mix(animated, corrected, envelope * vertex_mask)

  current = output
```

Important invariants:

- Sample 0 is always bind.
- Bind activation reapplies current animated detail, not a stored bind delta.
- RBF outputs must be projected onto the probability simplex. Do not replace this
  with simple clamping.
- The first implementation should use 16 representative vertices per area.
- `solveIterations` repeats relax -> solve -> apply for sharper convergence.
- Geometry comes first. Dynamic normal-map blending is a later feature.

## Blender Product Architecture

### Add-On Layer

Python owns Blender integration:

- Add-on registration.
- UI panels and operators.
- Source/bind/sample object references.
- Vertex group selection for deformation areas and mask.
- Training operator.
- Frame/depsgraph update handler.
- Reading evaluated source mesh positions.
- Writing final positions to the generated output mesh.
- Saving/loading configuration and training cache metadata.

Python must not do per-vertex deformation math in the live path. It should do
only glue operations and bulk array transfer. If the compiled extension is
unavailable, runtime training/evaluation should fail with a clear error instead
of using a Python numerical fallback.

### C++ Core Layer

The compiled extension owns performance-sensitive math:

- Mesh topology adjacency.
- Laplacian relaxation.
- Deformation-area CSR packing.
- Representative vertex sampling.
- Procrustes alignment.
- RBF training.
- RBF evaluation.
- Simplex projection.
- Corrective delta application.

Expose a small, stable API to Python. Keep Blender API calls out of C++ unless
there is a strong reason; the extension should operate on plain arrays. Use
Eigen explicitly for SVD, matrix solves, Procrustes, and vectorized linear
algebra instead of reimplementing fragile numerics or moving the work to Python.

Suggested C++ API:

```cpp
struct PoseTrainerSettings {
  int relaxIterations = 10;
  int solveIterations = 1;
  float rbfRadius = 0.1f;
  float regularization = 0.001f;
};

struct PoseTrainerCache {
  int vertexCount = 0;
  std::vector<std::vector<int>> neighbors;
  CSR areaWeights;
  std::vector<uint32_t> repIndices;
  std::vector<Vec3> bindRelaxed;
  std::vector<std::vector<Vec3>> sampleRelaxed;
  std::vector<std::vector<Vec3>> sampleDeltas;
  std::vector<RbfAreaModel> areas;
};

PoseTrainerCache train(
  const MeshTopology& topology,
  Span<Vec3> bind,
  Span<Span<Vec3>> samples,
  const CSR& areaWeights,
  const PoseTrainerSettings& settings
);

void evaluate(
  const PoseTrainerCache& cache,
  Span<Vec3> animated,
  Span<float> vertexMask,
  float envelope,
  MutableSpan<Vec3> out
);
```

## Blender Object Model

Use an output-object workflow:

1. Source object: user rig/modifier stack. Never mutate its mesh data during live
   updates.
2. Bind object: mesh object used as the bind sample.
3. Sample objects: corrective mesh objects.
4. Output object: generated mesh object receiving Pose Trainer result.

Blender mapping:

- Deformation areas: vertex groups on the source object or a configured area
  owner object.
- Deformer mask: one vertex group.
- Envelope/settings: add-on properties.
- Training cache: binary sidecar file or packed custom property blob.
- Output mesh: generated and updated by the add-on.

When reading the source, use the evaluated dependency graph so upstream Blender
deformers, armatures, shape keys, and animation are included.

## Packaging Direction

Users should install this as a normal Blender add-on.

Expected binary artifacts:

- Windows: `.pyd`
- Linux: `.so`
- macOS: `.so` or `.dylib` depending on packaging

The add-on may include platform-specific binaries and load the matching one at
runtime. Any Python fallback must be limited to explicitly marked diagnostics or
unit tests. It must not be used by the Blender add-on for training or live
deformation.

## Training Cache Rules

Invalidate or require retraining when any of these change:

- Source topology.
- Bind object.
- Sample object list.
- Sample object topology or vertex count.
- Deformation area vertex groups.
- Representative vertex sampling settings.
- Relax iterations.
- RBF radius.
- Regularization.

Changing envelope, mask weights, and solve iterations should not require full
RBF retraining unless the implementation makes them part of the cache.

## Numerical Requirements

- Operate in object-local space unless the user requests world-space behavior.
- Enforce matching vertex count and order for source, bind, and samples.
- Use stable Procrustes alignment with reflection handling.
- Use ridge regularization for RBF training.
- Project activation weights onto the probability simplex.
- Keep bind/sample/current features in the same aligned and scaled feature space.
- Avoid silently producing results if cache topology does not match the current
  evaluated mesh.

## Development Priorities

Build in this order:

1. Minimal add-on shell: source, bind, sample list, output object.
2. Python data-flow shell that marshals arrays only; no Python deformation
   backend in runtime.
3. C++ extension with train/evaluate API and Eigen-based linear algebra
   implementation.
4. Training cache save/load.
5. Live frame/depsgraph update.
6. Vertex groups for areas and mask.
7. Bake/export tools.
8. Performance tuning and parallelism.
9. Optional normal-map/material features.

## Anti-Patterns

- Do not require users to install a custom Blender build.
- Do not mutate the source object mesh during live updates.
- Do not do live deformation or training math in Python loops or Python numeric
  libraries.
- Do not silently use a Python fallback when the C++ extension is missing.
- Do not add new hand-rolled SVD, matrix inverse, or linear solve code when
  Eigen has the needed primitive.
- Do not treat clamp-and-normalize as equivalent to simplex projection.
- Do not ignore topology/vertex-order validation.
- Do not couple the C++ core directly to Blender data structures unless needed.
- Do not start with normal-map blending before geometry deformation is stable.
- Do not edit the Mush3D-3 reference repo unless explicitly asked.

## Agent Rules

1. Read existing files before editing.
2. Search for existing implementation patterns before adding new code.
3. Keep changes scoped to the current request.
4. Preserve user changes in the working tree.
5. Prefer clear, boring APIs between Python and C++.
6. Keep the C++ core array-based, fast, and testable outside Blender.
7. Validate topology and cache compatibility aggressively.
8. Add focused tests for math code when possible.
9. When uncertain, preserve the Mush3D Pose Trainer algorithm before optimizing
   or changing product behavior.
