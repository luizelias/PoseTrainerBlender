from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import bpy
import numpy as np
from bpy.app.handlers import persistent

from .core_loader import load_core


_CACHE_BY_SCENE_ID: dict[int, object] = {}
_HANDLER_RUNNING = False


@dataclass
class MeshSnapshot:
    positions: np.ndarray
    faces: list[list[int]]


def _scene_key(scene: bpy.types.Scene) -> int:
    return scene.as_pointer()


def _evaluated_mesh_snapshot(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> MeshSnapshot:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        positions = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", positions)
        faces = [list(poly.vertices) for poly in mesh.polygons]
        return MeshSnapshot(positions.reshape((-1, 3)), faces)
    finally:
        evaluated.to_mesh_clear()


def _vertex_group_weights(obj: bpy.types.Object, group_name: str, vertex_count: int, default: float) -> np.ndarray:
    weights = np.full(vertex_count, default, dtype=np.float32)
    if not group_name:
        return weights
    group = obj.vertex_groups.get(group_name)
    if group is None:
        raise ValueError(f'Vertex group "{group_name}" was not found on {obj.name}')

    weights.fill(0.0)
    for vertex in obj.data.vertices:
        if vertex.index >= vertex_count:
            continue
        for assignment in vertex.groups:
            if assignment.group == group.index:
                weights[vertex.index] = assignment.weight
                break
    return weights


def _collect_area_weights(settings, vertex_count: int) -> list[dict]:
    source = settings.source_object
    areas = []
    for area in settings.areas:
        if not area.group_name:
            continue
        areas.append({
            "name": area.group_name,
            "weights": _vertex_group_weights(source, area.group_name, vertex_count, 0.0),
        })
    if not areas:
        raise ValueError("Add at least one deformation area vertex group")
    return areas


def _ensure_output(settings, snapshot: MeshSnapshot) -> bpy.types.Object:
    source = settings.source_object
    output = settings.output_object
    if output is None or output.type != "MESH":
        mesh = bpy.data.meshes.new(f"{source.name}_PoseTrainerMesh")
        output = bpy.data.objects.new(f"{source.name}_PoseTrainer", mesh)
        source.users_collection[0].objects.link(output)
        settings.output_object = output

    if len(output.data.vertices) != len(snapshot.positions) or len(output.data.polygons) != len(snapshot.faces):
        mesh = bpy.data.meshes.new(f"{output.name}Mesh")
        mesh.from_pydata(snapshot.positions.tolist(), [], snapshot.faces)
        mesh.update()
        old_mesh = output.data
        output.data = mesh
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    return output


def _write_output_mesh(output: bpy.types.Object, positions: np.ndarray) -> None:
    mesh = output.data
    if len(mesh.vertices) != len(positions):
        raise ValueError("Output mesh topology does not match evaluated source")
    mesh.vertices.foreach_set("co", np.asarray(positions, dtype=np.float32).reshape(-1))
    mesh.update()


def create_output(context: bpy.types.Context) -> bpy.types.Object:
    settings = context.scene.pose_trainer
    if settings.source_object is None:
        raise ValueError("Choose a source object first")
    snapshot = _evaluated_mesh_snapshot(settings.source_object, context.evaluated_depsgraph_get())
    return _ensure_output(settings, snapshot)


def train_scene(context: bpy.types.Context) -> None:
    settings = context.scene.pose_trainer
    if settings.source_object is None:
        raise ValueError("Choose a source object")
    if settings.bind_object is None:
        raise ValueError("Choose a bind object")
    if len(settings.samples) == 0:
        raise ValueError("Add at least one corrective sample")

    depsgraph = context.evaluated_depsgraph_get()
    source_snapshot = _evaluated_mesh_snapshot(settings.source_object, depsgraph)
    bind_snapshot = _evaluated_mesh_snapshot(settings.bind_object, depsgraph)
    vertex_count = len(source_snapshot.positions)

    if len(bind_snapshot.positions) != vertex_count:
        raise ValueError("Bind object vertex count does not match source")

    samples = []
    for item in settings.samples:
        if item.object is None:
            continue
        sample = _evaluated_mesh_snapshot(item.object, depsgraph)
        if len(sample.positions) != vertex_count:
            raise ValueError(f"Sample {item.object.name} vertex count does not match source")
        samples.append(sample.positions)
    if not samples:
        raise ValueError("No valid sample objects are configured")

    areas = _collect_area_weights(settings, vertex_count)
    core = load_core()
    core_settings = core.PoseTrainerSettings()
    core_settings.relax_iterations = settings.relax_iterations
    if not hasattr(core_settings, "area_relax_iterations"):
        raise RuntimeError(
            "Installed Pose Trainer C++ core is outdated. Close Blender completely, "
            "then reinstall dist/pose_trainer_blender_addon.zip so the new core binary is loaded."
        )
    core_settings.area_relax_iterations = settings.area_relax_iterations
    core_settings.solve_iterations = settings.solve_iterations
    core_settings.rbf_radius = settings.rbf_radius
    core_settings.regularization = settings.regularization

    cache = core.train(source_snapshot.faces, bind_snapshot.positions, samples, areas, core_settings)
    _CACHE_BY_SCENE_ID[_scene_key(context.scene)] = cache
    settings.trained = True
    settings.status = f"Trained {len(samples)} sample(s), {len(areas)} area(s) with Eigen C++ core"
    _ensure_output(settings, source_snapshot)


def _evaluate_scene_with_depsgraph(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph) -> None:
    settings = scene.pose_trainer
    cache = _CACHE_BY_SCENE_ID.get(_scene_key(scene))
    if cache is None:
        raise ValueError("Train before evaluating")
    if settings.source_object is None:
        raise ValueError("Choose a source object")

    source_snapshot = _evaluated_mesh_snapshot(settings.source_object, depsgraph)
    output = _ensure_output(settings, source_snapshot)
    mask = _vertex_group_weights(settings.source_object, settings.mask_group, len(source_snapshot.positions), 1.0)
    positions = cache.evaluate(source_snapshot.positions, mask, settings.envelope, settings.solve_iterations)
    _write_output_mesh(output, positions)


def evaluate_scene(context: bpy.types.Context) -> None:
    _evaluate_scene_with_depsgraph(context.scene, context.evaluated_depsgraph_get())


def _live_update_scene(scene: bpy.types.Scene, depsgraph: Optional[bpy.types.Depsgraph] = None) -> None:
    global _HANDLER_RUNNING
    if _HANDLER_RUNNING:
        return
    settings = getattr(scene, "pose_trainer", None)
    if settings is None or not settings.live_update or not settings.trained:
        return
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    try:
        _HANDLER_RUNNING = True
        _evaluate_scene_with_depsgraph(scene, depsgraph)
    except Exception as exc:
        settings.status = str(exc)
    finally:
        _HANDLER_RUNNING = False


@persistent
def _depsgraph_handler(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph) -> None:
    _live_update_scene(scene, depsgraph)


@persistent
def _frame_change_handler(scene: bpy.types.Scene, depsgraph: Optional[bpy.types.Depsgraph] = None) -> None:
    _live_update_scene(scene, depsgraph)


def register_handlers() -> None:
    if _depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_handler)
    if _frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_frame_change_handler)


def unregister_handlers() -> None:
    if _depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_handler)
    if _frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_frame_change_handler)
    _CACHE_BY_SCENE_ID.clear()
