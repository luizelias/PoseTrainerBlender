import colorsys

import bpy

from .auto_masking import generate_auto_mask_areas
from . import runtime
from .properties import get_settings


AUTO_MASK_GROUP_PREFIX = "PT_AutoMask_"
AUTO_MASK_PREVIEW_ATTRIBUTE = "PT_AutoMaskPreview"
AUTO_MASK_PREVIEW_MATERIAL = "PT_AutoMaskPreview"
AUTO_MASK_PREVIEW_SLOT_PREFIX = "PT_AutoMaskPreview_"
AUTO_MASK_PREVIEW_SUFFIX = "_AutoMaskPreview"
UV_SHELL_GROUP_PREFIX = "PT_UVShell_"

_AUTO_MASK_BASE_COLORS = (
    (0.92, 0.12, 0.16),
    (0.08, 0.48, 0.95),
    (0.10, 0.72, 0.28),
    (1.00, 0.78, 0.08),
    (0.85, 0.22, 0.82),
    (0.00, 0.74, 0.72),
    (1.00, 0.45, 0.12),
    (0.45, 0.34, 0.95),
    (0.60, 0.86, 0.18),
    (0.95, 0.28, 0.48),
    (0.12, 0.64, 0.82),
    (0.78, 0.55, 0.08),
)


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


def _remove_generated_groups(settings, source: bpy.types.Object, prefix: str) -> int:
    generated_names = {group.name for group in source.vertex_groups if group.name.startswith(prefix)}
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


def _mesh_positions(mesh: bpy.types.Mesh) -> list[tuple[float, float, float]]:
    return [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in mesh.vertices]


def _mesh_faces(mesh: bpy.types.Mesh) -> list[list[int]]:
    return [list(poly.vertices) for poly in mesh.polygons if len(poly.vertices) >= 3]


def _create_weighted_vertex_group(source: bpy.types.Object, group_name: str, weights: dict[int, float]) -> None:
    group = source.vertex_groups.get(group_name)
    if group is not None:
        source.vertex_groups.remove(group)
    group = source.vertex_groups.new(name=group_name)
    for vertex_index, weight in sorted(weights.items()):
        group.add([vertex_index], float(weight), "REPLACE")


def _area_preview_color(index: int) -> tuple[float, float, float]:
    if index < len(_AUTO_MASK_BASE_COLORS):
        return _AUTO_MASK_BASE_COLORS[index]
    hue = (index * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.68, 0.95)


def _vertex_group_weight(source: bpy.types.Object, vertex: bpy.types.MeshVertex, group_index: int) -> float:
    for assignment in vertex.groups:
        if assignment.group == group_index:
            return float(assignment.weight)
    return 0.0


def _auto_mask_area_groups(settings, source: bpy.types.Object) -> list[bpy.types.VertexGroup]:
    groups = []
    for area in settings.areas:
        if not area.group_name.startswith(AUTO_MASK_GROUP_PREFIX):
            continue
        group = source.vertex_groups.get(area.group_name)
        if group is not None:
            groups.append(group)
    return groups


def _preview_vertex_colors(source: bpy.types.Object, area_groups: list[bpy.types.VertexGroup]) -> list[tuple[float, float, float, float]]:
    colors = []
    uncovered_color = (0.18, 0.18, 0.18)
    for vertex in source.data.vertices:
        mixed = [0.0, 0.0, 0.0]
        total = 0.0
        for area_index, group in enumerate(area_groups):
            weight = _vertex_group_weight(source, vertex, group.index)
            if weight <= 0.0:
                continue
            color = _area_preview_color(area_index)
            mixed[0] += color[0] * weight
            mixed[1] += color[1] * weight
            mixed[2] += color[2] * weight
            total += weight

        if total > 1.0:
            mixed = [channel / total for channel in mixed]
        else:
            uncovered = 1.0 - total
            mixed[0] += uncovered_color[0] * uncovered
            mixed[1] += uncovered_color[1] * uncovered
            mixed[2] += uncovered_color[2] * uncovered
        colors.append((mixed[0], mixed[1], mixed[2], 1.0))
    return colors


def _ensure_color_attribute(mesh: bpy.types.Mesh, name: str):
    color_attribute = mesh.color_attributes.get(name)
    if color_attribute is not None and (
        color_attribute.domain != "CORNER" or color_attribute.data_type != "BYTE_COLOR"
    ):
        mesh.color_attributes.remove(color_attribute)
        color_attribute = None
    if color_attribute is None:
        color_attribute = mesh.color_attributes.new(name=name, type="BYTE_COLOR", domain="CORNER")
    try:
        mesh.color_attributes.active_color = color_attribute
    except Exception:
        pass
    return color_attribute


def _ensure_preview_material(attribute_name: str) -> bpy.types.Material:
    material = bpy.data.materials.get(AUTO_MASK_PREVIEW_MATERIAL)
    if material is None:
        material = bpy.data.materials.new(AUTO_MASK_PREVIEW_MATERIAL)
    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        principled = nodes.new(type="ShaderNodeBsdfPrincipled")

    attribute = next(
        (
            node
            for node in nodes
            if node.type == "ATTRIBUTE" and getattr(node, "attribute_name", "") == attribute_name
        ),
        None,
    )
    if attribute is None:
        attribute = nodes.new(type="ShaderNodeAttribute")
        attribute.attribute_name = attribute_name
        attribute.location = (principled.location.x - 260, principled.location.y + 80)

    base_color = principled.inputs.get("Base Color")
    color_output = attribute.outputs.get("Color")
    if base_color is not None and color_output is not None:
        if not any(link.from_socket == color_output and link.to_socket == base_color for link in links):
            links.new(color_output, base_color)
    return material


def _ensure_flat_preview_material(index: int) -> bpy.types.Material:
    color = _area_preview_color(index)
    material_name = f"{AUTO_MASK_PREVIEW_SLOT_PREFIX}{index + 1:03d}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
    material.diffuse_color = (color[0], color[1], color[2], 1.0)
    material.use_nodes = True
    principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is not None:
        base_color = principled.inputs.get("Base Color")
        if base_color is not None:
            base_color.default_value = (color[0], color[1], color[2], 1.0)
        roughness = principled.inputs.get("Roughness")
        if roughness is not None:
            roughness.default_value = 0.75
    return material


def _mesh_bounds_diagonal(mesh: bpy.types.Mesh) -> float:
    if not mesh.vertices:
        return 0.0
    min_xyz = [float(mesh.vertices[0].co[axis]) for axis in range(3)]
    max_xyz = list(min_xyz)
    for vertex in mesh.vertices:
        for axis in range(3):
            value = float(vertex.co[axis])
            min_xyz[axis] = min(min_xyz[axis], value)
            max_xyz[axis] = max(max_xyz[axis], value)
    dx = max_xyz[0] - min_xyz[0]
    dy = max_xyz[1] - min_xyz[1]
    dz = max_xyz[2] - min_xyz[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _copy_source_mesh_for_preview(source: bpy.types.Object) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(f"{source.name}{AUTO_MASK_PREVIEW_SUFFIX}Mesh")
    mesh.from_pydata(_mesh_positions(source.data), [], _mesh_faces(source.data))
    mesh.update()
    source.data.update()
    offset = max(_mesh_bounds_diagonal(source.data) * 0.0015, 0.001)
    for vertex in mesh.vertices:
        normal = source.data.vertices[vertex.index].normal
        vertex.co.x += normal.x * offset
        vertex.co.y += normal.y * offset
        vertex.co.z += normal.z * offset
    mesh.update()
    return mesh


def _preview_object_name(source: bpy.types.Object) -> str:
    return f"{source.name}{AUTO_MASK_PREVIEW_SUFFIX}"


def _dominant_area_index(
    source: bpy.types.Object,
    area_groups: list[bpy.types.VertexGroup],
    vertex_indices,
) -> int:
    totals = [0.0] * len(area_groups)
    for vertex_index in vertex_indices:
        vertex = source.data.vertices[vertex_index]
        for area_index, group in enumerate(area_groups):
            totals[area_index] += _vertex_group_weight(source, vertex, group.index)
    return max(range(len(area_groups)), key=lambda index: (totals[index], -index))


def _remove_preview_object(source: bpy.types.Object) -> bool:
    preview = bpy.data.objects.get(_preview_object_name(source))
    if preview is None:
        return False
    mesh = preview.data if preview.type == "MESH" else None
    bpy.data.objects.remove(preview, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    return True


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
            _remove_generated_groups(settings, source, UV_SHELL_GROUP_PREFIX)

        existing_area_names = {area.group_name for area in settings.areas}
        created_count = 0
        for shell_index, weights in enumerate(shell_weights, start=1):
            group_name = f"{UV_SHELL_GROUP_PREFIX}{shell_index:03d}"
            _create_weighted_vertex_group(source, group_name, weights)

            if group_name not in existing_area_names:
                area = settings.areas.add()
                area.group_name = group_name
                existing_area_names.add(group_name)
            created_count += 1

        settings.area_index = min(settings.area_index, max(0, len(settings.areas) - 1))
        settings.trained = False
        settings.status = f"Created {created_count} UV shell area(s) from {source.name}"
        return {"FINISHED"}


class PT_OT_auto_mask(bpy.types.Operator):
    bl_idname = "pose_trainer.auto_mask"
    bl_label = "Auto Mask"
    bl_description = "Create stable local deformation-area vertex groups from mesh topology"
    bl_options = {"REGISTER", "UNDO"}

    replace_existing: bpy.props.BoolProperty(
        name="Replace Generated Auto Masks",
        default=True,
        description="Remove existing PT_AutoMask vertex groups and Pose Trainer areas before creating new ones",
    )

    def execute(self, context):
        settings = get_settings(context)
        source = settings.source_object or context.object
        if source is None or source.type != "MESH":
            self.report({"ERROR"}, "Choose a source mesh")
            return {"CANCELLED"}

        settings.source_object = source
        mesh = source.data
        positions = _mesh_positions(mesh)
        faces = _mesh_faces(mesh)
        if not positions or not faces:
            message = f'Mesh "{mesh.name}" needs vertices and faces for Auto Mask'
            settings.status = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        try:
            auto_areas = generate_auto_mask_areas(
                faces,
                positions,
                area_count=settings.auto_mask_area_count,
                softness_iterations=settings.auto_mask_softness,
            )
        except ValueError as exc:
            settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        if not auto_areas:
            message = f'Auto Mask found no usable areas on "{mesh.name}"'
            settings.status = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        if self.replace_existing:
            _remove_generated_groups(settings, source, AUTO_MASK_GROUP_PREFIX)

        existing_area_names = {area.group_name for area in settings.areas}
        for area_index, auto_area in enumerate(auto_areas, start=1):
            group_name = f"{AUTO_MASK_GROUP_PREFIX}{area_index:03d}"
            _create_weighted_vertex_group(source, group_name, auto_area.weights)
            if group_name not in existing_area_names:
                area = settings.areas.add()
                area.group_name = group_name
                existing_area_names.add(group_name)

        settings.area_index = min(settings.area_index, max(0, len(settings.areas) - 1))
        settings.trained = False
        settings.status = f"Created {len(auto_areas)} auto mask area(s) from {source.name}"
        return {"FINISHED"}


class PT_OT_preview_auto_mask(bpy.types.Operator):
    bl_idname = "pose_trainer.preview_auto_mask"
    bl_label = "Preview Auto Mask"
    bl_description = "Create or update a colored mesh preview of generated auto-mask areas"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        source = settings.source_object or context.object
        if source is None or source.type != "MESH":
            self.report({"ERROR"}, "Choose a source mesh")
            return {"CANCELLED"}

        area_groups = _auto_mask_area_groups(settings, source)
        if not area_groups:
            message = "Create Auto Mask areas before previewing them"
            settings.status = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        preview_name = _preview_object_name(source)
        preview = bpy.data.objects.get(preview_name)
        needs_mesh = (
            preview is None
            or preview.type != "MESH"
            or len(preview.data.vertices) != len(source.data.vertices)
            or len(preview.data.polygons) != len(source.data.polygons)
        )
        if preview is None or preview.type != "MESH":
            preview_mesh = _copy_source_mesh_for_preview(source)
            preview = bpy.data.objects.new(preview_name, preview_mesh)
            collection = source.users_collection[0] if source.users_collection else context.scene.collection
            collection.objects.link(preview)
        elif needs_mesh:
            old_mesh = preview.data
            preview.data = _copy_source_mesh_for_preview(source)
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)

        preview.matrix_world = source.matrix_world
        preview.display_type = "TEXTURED"
        preview.hide_render = True
        preview.show_in_front = True

        material = _ensure_preview_material(AUTO_MASK_PREVIEW_ATTRIBUTE)
        preview.data.materials.clear()
        preview.data.materials.append(material)
        for area_index in range(len(area_groups)):
            preview.data.materials.append(_ensure_flat_preview_material(area_index))
        for poly in preview.data.polygons:
            poly.material_index = 1 + _dominant_area_index(source, area_groups, poly.vertices)

        vertex_colors = _preview_vertex_colors(source, area_groups)
        color_attribute = _ensure_color_attribute(preview.data, AUTO_MASK_PREVIEW_ATTRIBUTE)
        for loop_index, loop in enumerate(preview.data.loops):
            color_attribute.data[loop_index].color = vertex_colors[loop.vertex_index]
        preview.data.update()

        for area in context.screen.areas if context.screen else []:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "MATERIAL"
                    break

        for obj in context.selected_objects:
            obj.select_set(False)
        preview.select_set(True)
        context.view_layer.objects.active = preview

        settings.status = f"Updated auto mask preview: {preview.name}"
        return {"FINISHED"}


class PT_OT_clear_masking(bpy.types.Operator):
    bl_idname = "pose_trainer.clear_masking"
    bl_label = "Clear Masking"
    bl_description = "Remove Pose Trainer area assignments, generated mask groups, and auto-mask preview"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        source = settings.source_object or context.object
        removed_groups = 0
        removed_preview = False

        if source is not None and source.type == "MESH":
            generated_names = {
                group.name
                for group in source.vertex_groups
                if group.name.startswith(AUTO_MASK_GROUP_PREFIX) or group.name.startswith(UV_SHELL_GROUP_PREFIX)
            }
            for group_name in sorted(generated_names, reverse=True):
                group = source.vertex_groups.get(group_name)
                if group is not None:
                    source.vertex_groups.remove(group)
                    removed_groups += 1
            removed_preview = _remove_preview_object(source)

        removed_areas = len(settings.areas)
        for index in range(len(settings.areas) - 1, -1, -1):
            settings.areas.remove(index)
        settings.area_index = 0
        settings.mask_group = ""
        settings.trained = False

        suffix = " and preview" if removed_preview else ""
        settings.status = f"Cleared {removed_areas} area(s), {removed_groups} generated group(s){suffix}"
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
