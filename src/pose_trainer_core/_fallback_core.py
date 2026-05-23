from __future__ import annotations

from dataclasses import dataclass


_ERROR = (
    "Pose Trainer requires the compiled Eigen-based C++ extension "
    "_pose_trainer_core for training and evaluation."
)


@dataclass
class PoseTrainerSettings:
    relax_iterations: int = 10
    solve_iterations: int = 1
    rbf_radius: float = 0.1
    regularization: float = 0.001


class PoseTrainerCache:
    vertex_count = 0
    last_backend = "Unavailable"
    last_opencl_timing = ""

    def evaluate(self, *_args, **_kwargs):
        raise RuntimeError(_ERROR)


def train(*_args, **_kwargs) -> PoseTrainerCache:
    raise RuntimeError(_ERROR)


def project_simplex(*_args, **_kwargs):
    raise RuntimeError(_ERROR)


def opencl_available() -> bool:
    return False


def opencl_status() -> str:
    return _ERROR
