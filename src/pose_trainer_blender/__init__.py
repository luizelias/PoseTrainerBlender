bl_info = {
    "name": "Pose Trainer Blender",
    "author": "Codex",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Pose Trainer",
    "description": "Example-trained pose-space Delta Mush deformer using a C++ Python core",
    "category": "Animation",
}

from . import operators, properties, runtime, ui


CLASSES = (
    properties.PT_SampleItem,
    properties.PT_AreaItem,
    properties.PT_Settings,
    operators.PT_OT_add_selected_samples,
    operators.PT_OT_remove_sample,
    operators.PT_OT_add_active_area_group,
    operators.PT_OT_remove_area,
    operators.PT_OT_extract_areas_from_uv_shells,
    operators.PT_OT_create_output,
    operators.PT_OT_train,
    operators.PT_OT_evaluate_once,
    operators.PT_OT_copy_profile_timing,
    operators.PT_OT_toggle_live_update,
    ui.PT_UL_samples,
    ui.PT_UL_areas,
    ui.PT_PT_panel,
)


def register():
    import bpy

    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pose_trainer = bpy.props.PointerProperty(type=properties.PT_Settings)
    runtime.register_handlers()


def unregister():
    import bpy

    runtime.unregister_handlers()
    if hasattr(bpy.types.Scene, "pose_trainer"):
        del bpy.types.Scene.pose_trainer
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
