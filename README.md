# Blender Pose Trainer

Blender-friendly example-trained pose-space Delta Mush deformer inspired by the
Mush3D Pose Trainer reference implementation.

This repository is intentionally shaped as a normal Blender add-on plus a
compiled Python extension:

- `src/pose_trainer_blender`: Blender UI, operators, depsgraph integration, and
  output mesh updates.
- `src/pose_trainer_core`: Python package that imports the compiled
  `_pose_trainer_core` extension. The Blender add-on requires the compiled core
  for training/evaluation.
- `src/pose_trainer_core/cpp`: array-based C++ core; no Blender API dependency.
  Production deformation math belongs here and uses Eigen explicitly for
  Procrustes, SVD, and RBF linear solves.
- `tests`: focused math and API contract tests.

The `REFERENCE` directory is the algorithm reference and should remain read-only.

## Current Add-On Features

- Source, bind, corrective sample, output, area, mask, and solver settings in
  `View3D > Sidebar > Pose Trainer`.
- Output-object workflow: the source object is not mutated during evaluation.
- C++/Eigen training and evaluation through the compiled Python extension.
- Optional OpenCL runtime backend for live evaluation. The current OpenCL path
  accelerates Mush3D-style half-edge mesh relaxation, per-area RBF/Procrustes
  evaluation, and final per-vertex delta application, then reads final positions
  back for Blender.
  Its command queue avoids intermediate hard waits between kernels, reuses the
  mask buffer when possible, and reports `GPU wall` plus
  upload/relax/area/apply/read execution buckets through the add-on's
  `Profile Timing` display.
- UV shell extraction for area masks: use `Areas From UV Shells` on a source
  mesh with an active UV map to create `PT_UVShell_###` vertex groups and add
  them as Pose Trainer deformation areas.
- Auto Mask generation: use `Auto Mask` to create deterministic
  `PT_AutoMask_###` vertex groups from topology-aware local clustering, with
  compact softened boundaries suitable for Pose Trainer deformation areas.
- Auto Mask preview: use `Preview Auto Mask` to create a non-rendering colored
  preview mesh that displays the generated regions and blended boundaries.
- Clear generated masking with `Clear Masking`, which removes Pose Trainer area
  assignments, generated Auto Mask/UV shell groups, and the preview object.

## Build

```powershell
python -m pip install -e .[test]
python -m pytest
```

On Windows, building the extension requires a C++17 compiler compatible with the
Python interpreter used by Blender or your local test Python.

## Blender Add-On

Create an installable add-on zip:

```powershell
python tools/package_addon.py
```

Then install `dist/pose_trainer_blender_addon.zip` in Blender via
`Edit > Preferences > Add-ons > Install...`.

When `build_support/wheel_cp313/pose_trainer_core` exists, the package script
bundles the Blender 5.1 / Python 3.13 `_pose_trainer_core` binary inside the
add-on zip. The current local `dist/pose_trainer_blender_addon.zip` includes
that binary and has been smoke-tested in Blender 5.1.
