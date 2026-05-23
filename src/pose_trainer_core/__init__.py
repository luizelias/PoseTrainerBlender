"""Pose Trainer core Python package.

The compiled Eigen C++ extension is required for training and evaluation. The
fallback only preserves importability and raises a clear build/install error.
"""

try:
    from ._pose_trainer_core import (
        PoseTrainerCache,
        PoseTrainerSettings,
        opencl_available,
        opencl_status,
        project_simplex,
        train,
    )
    USING_FALLBACK = False
except Exception:  # pragma: no cover - exercised when no extension is present
    from ._fallback_core import (
        PoseTrainerCache,
        PoseTrainerSettings,
        opencl_available,
        opencl_status,
        project_simplex,
        train,
    )
    USING_FALLBACK = True

__all__ = [
    "PoseTrainerCache",
    "PoseTrainerSettings",
    "USING_FALLBACK",
    "opencl_available",
    "opencl_status",
    "project_simplex",
    "train",
]
