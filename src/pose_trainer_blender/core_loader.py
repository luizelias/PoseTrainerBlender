from __future__ import annotations

import importlib


def load_core():
    errors = []
    try:
        core = importlib.import_module(".pose_trainer_core", __package__)
    except ModuleNotFoundError as exc:
        if exc.name not in {f"{__package__}.pose_trainer_core", "pose_trainer_core"}:
            raise
        errors.append(exc)
        try:
            core = importlib.import_module("pose_trainer_core")
        except ModuleNotFoundError as top_level_exc:
            if top_level_exc.name != "pose_trainer_core":
                raise
            errors.append(top_level_exc)
            raise RuntimeError(
                "Pose Trainer UI is installed, but the compiled Eigen C++ core is "
                "not installed for this Blender Python. Build/package "
                "_pose_trainer_core for Blender before training or evaluating."
            ) from top_level_exc
    except ImportError as exc:
        errors.append(exc)
        try:
            core = importlib.import_module("pose_trainer_core")
        except Exception as top_level_exc:
            errors.append(top_level_exc)
            details = "; ".join(str(error) for error in errors if str(error))
            raise RuntimeError(
                "Pose Trainer could not load the compiled Eigen C++ core for "
                f"this Blender Python. Details: {details}"
            ) from top_level_exc

    if getattr(core, "USING_FALLBACK", False):
        try:
            core = importlib.import_module("pose_trainer_core")
        except Exception:
            pass
    if getattr(core, "USING_FALLBACK", False):
        raise RuntimeError(
            "Pose Trainer requires the compiled Eigen C++ core for Blender runtime "
            "training/evaluation. Build and install _pose_trainer_core for this "
            "Blender/Python platform."
        )
    return core
