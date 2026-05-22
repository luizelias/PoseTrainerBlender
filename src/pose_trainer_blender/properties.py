import bpy


def _refresh_output_after_setting_change(self, context):
    if context is None or not getattr(self, "trained", False):
        return
    try:
        from . import runtime

        runtime.evaluate_scene(context)
    except Exception as exc:
        self.status = str(exc)


def _mark_training_stale(self, _context):
    if getattr(self, "trained", False):
        self.trained = False
        self.status = "Training settings changed; retrain Pose Trainer"


class PT_SampleItem(bpy.types.PropertyGroup):
    object: bpy.props.PointerProperty(
        name="Sample",
        type=bpy.types.Object,
        poll=lambda _self, obj: obj.type == "MESH",
    )


class PT_AreaItem(bpy.types.PropertyGroup):
    group_name: bpy.props.StringProperty(name="Vertex Group")


class PT_Settings(bpy.types.PropertyGroup):
    source_object: bpy.props.PointerProperty(
        name="Source",
        type=bpy.types.Object,
        poll=lambda _self, obj: obj.type == "MESH",
    )
    bind_object: bpy.props.PointerProperty(
        name="Bind",
        type=bpy.types.Object,
        poll=lambda _self, obj: obj.type == "MESH",
    )
    output_object: bpy.props.PointerProperty(
        name="Output",
        type=bpy.types.Object,
        poll=lambda _self, obj: obj.type == "MESH",
    )

    samples: bpy.props.CollectionProperty(type=PT_SampleItem)
    sample_index: bpy.props.IntProperty(default=0)

    areas: bpy.props.CollectionProperty(type=PT_AreaItem)
    area_index: bpy.props.IntProperty(default=0)
    mask_group: bpy.props.StringProperty(name="Mask Group")

    relax_iterations: bpy.props.IntProperty(
        name="Relax",
        default=10,
        min=1,
        max=50,
        update=_mark_training_stale,
    )
    area_relax_iterations: bpy.props.IntProperty(
        name="Area Relax",
        default=0,
        min=0,
        max=50,
        description="Blur deformation area weights before C++ Pose Trainer training",
        update=_mark_training_stale,
    )
    solve_iterations: bpy.props.IntProperty(
        name="Solve",
        default=1,
        min=1,
        max=5,
        update=_refresh_output_after_setting_change,
    )
    rbf_radius: bpy.props.FloatProperty(
        name="RBF Radius",
        default=0.1,
        min=0.0001,
        update=_mark_training_stale,
    )
    regularization: bpy.props.FloatProperty(
        name="Regularization",
        default=0.001,
        min=0.0,
        update=_mark_training_stale,
    )
    envelope: bpy.props.FloatProperty(
        name="Envelope",
        default=1.0,
        min=0.0,
        max=1.0,
        description="Blend between the evaluated source mesh and the full Pose Trainer deformation",
        update=_refresh_output_after_setting_change,
    )

    live_update: bpy.props.BoolProperty(name="Live Update", default=False)
    trained: bpy.props.BoolProperty(name="Trained", default=False, options={"SKIP_SAVE"})
    status: bpy.props.StringProperty(name="Status", default="Not trained", options={"SKIP_SAVE"})


def get_settings(context: bpy.types.Context) -> PT_Settings:
    return context.scene.pose_trainer
