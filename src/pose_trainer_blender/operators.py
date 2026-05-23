import bpy

from . import runtime
from .properties import get_settings


UV_SHELL_GROUP_PREFIX = "PT_UVShell_"


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def unite(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _quantized_uv_key(uv, tolerance: float) -> tuple[int, int]:
    return (round(float(uv.x) / tolerance), round(float(uv.y) / tolerance))


def _mesh_edge_key(vertex_a: int, vertex_b: int) -> tuple[int, int]:
    return (vertex_a, vertex_b) if vertex_a <= vertex_b else (vertex_b, vertex_a)


def _uv_edge_key(uv_a, uv_b, tolerance: float) -> tuple[tuple[int, int], tuple[int, int]]:
    key_a = _quantized_uv_key(uv_a, tolerance)
    key_b = _quantized_uv_key(uv_b, tolerance)
    return (key_a, key_b) if key_a <= key_b else (key_b, key_a)


def _compute_uv_shell_vertex_weights(mesh: bpy.types.Mesh, tolerance: float = 1.0e-6) -> list[dict[int, float]]:
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise ValueError(f'Mesh "{mesh.name}" has no active UV map')

    polygons = [poly for poly in mesh.polygons if len(poly.vertices) >= 3]
    if not polygons:
        return []

    polygon_index_to_local = {poly.index: local for local, poly in enumerate(polygons)}
    union_find = _UnionFind(len(polygons))
    edge_to_faces: dict[tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], list[int]] = {}

    for poly in polygons:
        local_poly_index = polygon_index_to_local[poly.index]
        vertices = list(poly.vertices)
        loop_indices = list(poly.loop_indices)
        vertex_count = len(vertices)
        for corner in range(vertex_count):
            next_corner = (corner + 1) % vertex_count
            edge_key = _mesh_edge_key(vertices[corner], vertices[next_corner])
            uv_key = _uv_edge_key(
                uv_layer.data[loop_indices[corner]].uv,
                uv_layer.data[loop_indices[next_corner]].uv,
                tolerance,
            )
            full_key = (edge_key, uv_key)
            connected_faces = edge_to_faces.setdefault(full_key, [])
            for other_poly_index in connected_faces:
                union_find.unite(local_poly_index, other_poly_index)
            connected_faces.append(local_poly_index)

    root_to_shell: dict[int, int] = {}
    face_shell_ids = [0] * len(polygons)
    for local_poly_index, poly in enumerate(polygons):
        root = union_find.find(local_poly_index)
        shell_id = root_to_shell.setdefault(root, len(root_to_shell))
        face_shell_ids[local_poly_index] = shell_id

    shell_count = len(root_to_shell)
    vertex_shell_counts: list[dict[int, int]] = [dict() for _ in range(len(mesh.vertices))]
    for local_poly_index, poly in enumerate(polygons):
        shell_id = face_shell_ids[local_poly_index]
        for vertex_index in poly.vertices:
            counts = vertex_shell_counts[vertex_index]
            counts[shell_id] = counts.get(shell_id, 0) + 1

    shell_weights: list[dict[int, float]] = [dict() for _ in range(shell_count)]
    for vertex_index, counts in enumerate(vertex_shell_counts):
        total = sum(counts.values())
        if total <= 0:
            continue
        for shell_id, count in counts.items():
            shell_weights[shell_id][vertex_index] = count / total

    shell_weights.sort(key=lambda weights: min(weights.keys()) if weights else len(mesh.vertices))
    return shell_weights


def _remove_generated_uv_shell_groups(settings, source: bpy.types.Object) -> int:
    generated_names = {group.name for group in source.vertex_groups if group.name.startswith(UV_SHELL_GROUP_PREFIX)}
    if not generated_names:
        return 0

    for group_name in sorted(generated_names, reverse=True):
        group = source.vertex_groups.get(group_name)
        if group is not None:
            source.vertex_groups.remove(group)

    removed_areas = 0
    for index in range(len(settings.areas) - 1, -1, -1):
        if settings.areas[index].group_name in generated_names:
            settings.areas.remove(index)
            removed_areas += 1
    settings.area_index = min(settings.area_index, max(0, len(settings.areas) - 1))
    return removed_areas


class PT_OT_add_selected_samples(bpy.types.Operator):
    bl_idname = "pose_trainer.add_selected_samples"
    bl_label = "Add Selected Samples"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        existing = {item.object for item in settings.samples}
        for obj in context.selected_objects:
            if obj.type != "MESH" or obj in existing:
                continue
            if obj in {settings.source_object, settings.bind_object, settings.output_object}:
                continue
            item = settings.samples.add()
            item.object = obj
            existing.add(obj)
        settings.status = f"{len(settings.samples)} sample(s) configured"
        return {"FINISHED"}


class PT_OT_remove_sample(bpy.types.Operator):
    bl_idname = "pose_trainer.remove_sample"
    bl_label = "Remove Sample"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty()

    def execute(self, context):
        settings = get_settings(context)
        if 0 <= self.index < len(settings.samples):
            settings.samples.remove(self.index)
            settings.sample_index = min(settings.sample_index, max(0, len(settings.samples) - 1))
            settings.trained = False
        return {"FINISHED"}


class PT_OT_add_active_area_group(bpy.types.Operator):
    bl_idname = "pose_trainer.add_active_area_group"
    bl_label = "Add Active Area"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        source = settings.source_object or context.object
        if source is None or source.type != "MESH" or source.vertex_groups.active is None:
            self.report({"ERROR"}, "Select a source mesh with an active vertex group")
            return {"CANCELLED"}
        settings.source_object = source
        name = source.vertex_groups.active.name
        if all(area.group_name != name for area in settings.areas):
            item = settings.areas.add()
            item.group_name = name
            settings.trained = False
        return {"FINISHED"}


class PT_OT_remove_area(bpy.types.Operator):
    bl_idname = "pose_trainer.remove_area"
    bl_label = "Remove Area"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty()

    def execute(self, context):
        settings = get_settings(context)
        if 0 <= self.index < len(settings.areas):
            settings.areas.remove(self.index)
            settings.area_index = min(settings.area_index, max(0, len(settings.areas) - 1))
            settings.trained = False
        return {"FINISHED"}


class PT_OT_extract_areas_from_uv_shells(bpy.types.Operator):
    bl_idname = "pose_trainer.extract_areas_from_uv_shells"
    bl_label = "Areas From UV Shells"
    bl_options = {"REGISTER", "UNDO"}

    replace_existing: bpy.props.BoolProperty(
        name="Replace Generated Shell Areas",
        default=True,
        description="Remove existing PT_UVShell vertex groups and Pose Trainer areas before creating new ones",
    )

    def execute(self, context):
        settings = get_settings(context)
        source = settings.source_object or context.object
        if source is None or source.type != "MESH":
            self.report({"ERROR"}, "Choose a source mesh with an active UV map")
            return {"CANCELLED"}

        settings.source_object = source
        mesh = source.data
        try:
            shell_weights = _compute_uv_shell_vertex_weights(mesh)
        except ValueError as exc:
            settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        if not shell_weights:
            message = f'Mesh "{mesh.name}" has no UV shell faces'
            settings.status = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        if self.replace_existing:
            _remove_generated_uv_shell_groups(settings, source)

        existing_area_names = {area.group_name for area in settings.areas}
        created_count = 0
        for shell_index, weights in enumerate(shell_weights, start=1):
            group_name = f"{UV_SHELL_GROUP_PREFIX}{shell_index:03d}"
            group = source.vertex_groups.get(group_name)
            if group is not None:
                source.vertex_groups.remove(group)
            group = source.vertex_groups.new(name=group_name)

            for vertex_index, weight in weights.items():
                group.add([vertex_index], float(weight), "REPLACE")

            if group_name not in existing_area_names:
                area = settings.areas.add()
                area.group_name = group_name
                existing_area_names.add(group_name)
            created_count += 1

        settings.area_index = min(settings.area_index, max(0, len(settings.areas) - 1))
        settings.trained = False
        settings.status = f"Created {created_count} UV shell area(s) from {source.name}"
        return {"FINISHED"}


class PT_OT_create_output(bpy.types.Operator):
    bl_idname = "pose_trainer.create_output"
    bl_label = "Create Output"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        try:
            output = runtime.create_output(context)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        settings.status = f"Output object: {output.name}"
        return {"FINISHED"}


class PT_OT_train(bpy.types.Operator):
    bl_idname = "pose_trainer.train"
    bl_label = "Train Pose Trainer"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        try:
            runtime.train_scene(context)
            runtime.evaluate_scene(context)
        except Exception as exc:
            settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class PT_OT_evaluate_once(bpy.types.Operator):
    bl_idname = "pose_trainer.evaluate_once"
    bl_label = "Evaluate Once"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        try:
            runtime.evaluate_scene(context)
        except Exception as exc:
            settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class PT_OT_copy_profile_timing(bpy.types.Operator):
    bl_idname = "pose_trainer.copy_profile_timing"
    bl_label = "Copy Timing"
    bl_description = "Copy the latest Pose Trainer timing breakdown to the clipboard"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = get_settings(context)
        lines = []
        if settings.last_train_timing:
            lines.append(settings.last_train_timing)
        if settings.last_eval_timing:
            lines.append(settings.last_eval_timing)
        if getattr(settings, "last_eval_blender_timing", ""):
            lines.append(settings.last_eval_blender_timing)
        if getattr(settings, "last_eval_gpu_timing", ""):
            lines.append(settings.last_eval_gpu_timing)
        if not lines and settings.status:
            lines.append(settings.status)

        if not lines:
            self.report({"WARNING"}, "No Pose Trainer timing has been recorded yet")
            return {"CANCELLED"}

        context.window_manager.clipboard = "\n".join(lines)
        settings.status = "Copied Pose Trainer timing to clipboard"
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


class PT_OT_toggle_live_update(bpy.types.Operator):
    bl_idname = "pose_trainer.toggle_live_update"
    bl_label = "Toggle Live Update"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        settings.live_update = not settings.live_update
        settings.status = "Live update on" if settings.live_update else "Live update off"
        return {"FINISHED"}
