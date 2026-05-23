from __future__ import annotations

import bpy


class PT_UL_samples(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        name = item.object.name if item.object else "Missing sample"
        layout.label(text=name, icon="MESH_DATA")


class PT_UL_areas(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        layout.label(text=item.group_name or "Unnamed area", icon="GROUP_VERTEX")


class PT_PT_panel(bpy.types.Panel):
    bl_label = "Pose Trainer"
    bl_idname = "PT_PT_pose_trainer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pose Trainer"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.pose_trainer

        layout.prop(settings, "source_object")
        layout.prop(settings, "bind_object")
        layout.prop(settings, "output_object")
        layout.operator("pose_trainer.create_output", icon="MESH_DATA")

        layout.separator()
        row = layout.row()
        row.template_list("PT_UL_samples", "", settings, "samples", settings, "sample_index", rows=3)
        column = row.column(align=True)
        column.operator("pose_trainer.add_selected_samples", text="", icon="ADD")
        remove = column.operator("pose_trainer.remove_sample", text="", icon="REMOVE")
        remove.index = settings.sample_index

        layout.separator()
        row = layout.row()
        row.template_list("PT_UL_areas", "", settings, "areas", settings, "area_index", rows=3)
        column = row.column(align=True)
        column.operator("pose_trainer.add_active_area_group", text="", icon="ADD")
        remove = column.operator("pose_trainer.remove_area", text="", icon="REMOVE")
        remove.index = settings.area_index
        layout.operator("pose_trainer.extract_areas_from_uv_shells", icon="GROUP_VERTEX")
        if settings.source_object:
            layout.prop_search(settings, "mask_group", settings.source_object, "vertex_groups", text="Mask")
        else:
            layout.prop(settings, "mask_group", text="Mask")

        layout.separator()
        grid = layout.grid_flow(columns=2, even_columns=True)
        grid.prop(settings, "relax_iterations")
        grid.prop(settings, "area_relax_iterations")
        grid.prop(settings, "solve_iterations")
        grid.prop(settings, "runtime_backend")
        grid.prop(settings, "rbf_radius")
        grid.prop(settings, "regularization")
        layout.prop(settings, "envelope", slider=True)

        layout.separator()
        row = layout.row(align=True)
        row.operator("pose_trainer.train", icon="PLAY")
        row.operator("pose_trainer.evaluate_once", icon="FILE_REFRESH")
        icon = "PAUSE" if settings.live_update else "PLAY"
        layout.operator("pose_trainer.toggle_live_update", icon=icon, depress=settings.live_update)
        layout.prop(settings, "profile_timing")
        if settings.last_train_timing:
            layout.label(text=settings.last_train_timing)
        if settings.profile_timing and settings.last_eval_timing:
            layout.label(text=settings.last_eval_timing)
            if settings.last_eval_blender_timing:
                layout.label(text=settings.last_eval_blender_timing)
            if settings.last_eval_gpu_timing:
                layout.label(text=settings.last_eval_gpu_timing)
            layout.operator("pose_trainer.copy_profile_timing", text="Copy Timing")
        if not (settings.profile_timing and settings.last_eval_timing and settings.status.startswith(settings.last_eval_timing)):
            layout.label(text=settings.status)
