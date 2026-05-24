from __future__ import annotations

import bpy


class PT_UL_samples(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        name = item.object.name if item.object else "Missing sample"
        split = layout.split(factor=0.58)
        split.label(text=name, icon="MESH_DATA")
        bind_column = split.row(align=True)
        bind_column.prop(item, "is_bind_pose", text="Bind Mesh")


class PT_UL_areas(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        layout.label(text=item.group_name or "Unnamed area", icon="GROUP_VERTEX")


class PT_PT_panel(bpy.types.Panel):
    bl_label = "Pose Trainer"
    bl_idname = "PT_PT_pose_trainer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pose Trainer"

    def _section(self, layout, title, icon):
        box = layout.box()
        box.label(text=title, icon=icon)
        return box

    def draw(self, context):
        layout = self.layout
        settings = context.scene.pose_trainer

        layout.use_property_split = True
        layout.use_property_decorate = False

        setup = self._section(layout, "1. Source", "OUTLINER_OB_MESH")
        setup.prop(settings, "source_object")
        setup.prop(settings, "output_object", text="Result")

        samples = self._section(layout, "2. Training Meshes", "SHAPEKEY_DATA")
        row = samples.row()
        row.template_list("PT_UL_samples", "", settings, "samples", settings, "sample_index", rows=4)
        column = row.column(align=True)
        column.operator("pose_trainer.add_selected_samples", text="", icon="ADD")
        remove = column.operator("pose_trainer.remove_sample", text="", icon="REMOVE")
        remove.index = settings.sample_index

        train = self._section(layout, "3. Train", "PLAY")
        train.prop(settings, "envelope", slider=True)
        train.operator("pose_trainer.train", text="Train Pose Trainer", icon="PLAY")
        icon = "PAUSE" if settings.live_update else "PLAY"
        train.operator("pose_trainer.toggle_live_update", text="Live Update", icon=icon, depress=settings.live_update)

        advanced_row = layout.row()
        icon = "TRIA_DOWN" if settings.show_advanced else "TRIA_RIGHT"
        advanced_row.prop(settings, "show_advanced", text="Advanced", icon=icon, emboss=False)
        if settings.show_advanced:
            advanced = layout.box()
            grid = advanced.grid_flow(columns=2, even_columns=True)
            grid.prop(settings, "relax_iterations")
            grid.prop(settings, "area_relax_iterations")
            grid.prop(settings, "solve_iterations")
            grid.prop(settings, "runtime_backend")
            grid.prop(settings, "rbf_radius")
            grid.prop(settings, "auto_mask_area_count")
            grid.prop(settings, "auto_mask_softness")

        layout.separator()
        layout.label(text=settings.status, icon="INFO")
