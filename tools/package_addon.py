from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pose_trainer_blender"
CORE_SRC = ROOT / "src" / "pose_trainer_core"
WHEEL_CORE = ROOT / "build_support" / "wheel_cp313" / "pose_trainer_core"
DIST = ROOT / "dist"
ZIP_PATH = DIST / "pose_trainer_blender_addon.zip"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing add-on source: {SRC}")
    DIST.mkdir(exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SRC.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            if path.suffix in {".pyc", ".pyd", ".so", ".dylib"}:
                continue
            archive.write(path, Path("pose_trainer_blender") / path.relative_to(SRC))
        core_source = WHEEL_CORE if WHEEL_CORE.exists() else CORE_SRC
        if not any(core_source.glob("_pose_trainer_core*.pyd")):
            print("warning: packaging add-on without compiled Blender core")
        for path in sorted(core_source.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or "cpp" in path.parts:
                continue
            if path.suffix in {".pyc", ".so", ".dylib"}:
                continue
            archive.write(
                path,
                Path("pose_trainer_blender") / "pose_trainer_core" / path.relative_to(core_source),
            )

    print(ZIP_PATH)


if __name__ == "__main__":
    main()
