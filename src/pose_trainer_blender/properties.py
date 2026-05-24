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


def _mark_sample_training_stale(context):
    if context is None or getattr(context, "scene", None) is None:
        return
    settings = getattr(context.scene, "pose_trainer", None)
    if settings is None:
        return
    if getattr(settings, "trained", False):
        settings.trained = False
        settings.status = "Sample settings changed; retrain Pose Trainer"


def _sample_changed(_self, context):
    _mark_sample_training_stale(context)


def _sample_bind_pose_changed(self, context):
    if context is None or getattr(context, "scene", None) is None:
        return
    settings = getattr(context.scene, "pose_trainer", None)
    if settings is None:
        return
    if self.is_bind_pose:
        self_pointer = self.as_pointer()
        for item in settings.samples:
            if item.as_pointer() != self_pointer and item.is_bind_pose:
                item.is_bind_pose = False
    _mark_sample_training_stale(context)


class PT_SampleItem(bpy.types.PropertyGroup):
    object: bpy.props.PointerProperty(
        name="Sample",
        type=bpy.types.Object,
        poll=lambda _self, obj: obj.type == "MESH",
        update=_sample_changed,
    )
    is_bind_pose: bpy.props.BoolProperty(
        name="Bind Mesh",
        description="Use this mesh as the bind/rest pose reference for Pose Trainer training",
        default=False,
        update=_sample_bind_pose_changed,
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
    auto_mask_area_count: bpy.props.IntProperty(
        name="Auto Areas",
        default=32,
        min=1,
        max=512,
        description="Target number of generated local deformation areas",
    )
    auto_mask_softness: bpy.props.IntProperty(
        name="Auto Softness",
        default=2,
        min=0,
        max=8,
        description="Number of local boundary smoothing passes for generated auto masks",
    )

    relax_iterations: bpy.props.IntProperty(
        name="Mesh Relax",
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
        name="Solve Iterations",
        default=1,
        min=1,
        max=5,
        update=_refresh_output_after_setting_change,
    )
    runtime_backend: bpy.props.EnumProperty(
        name="Backend",
        items=(
            ("0", "Auto", "Use OpenCL when available, otherwise use CPU"),
            ("1", "CPU", "Use the Eigen C++ CPU evaluator"),
            ("2", "OpenCL", "Require the OpenCL runtime evaluator"),
        ),
        default="0",
        update=_refresh_output_after_setting_change,
    )
    rbf_radius: bpy.props.FloatProperty(
        name="Activation Radius",
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

    show_advanced: bpy.props.BoolProperty(name="Advanced", default=False)
    live_update: bpy.props.BoolProperty(name="Live Update", default=False)
    profile_timing: bpy.props.BoolProperty(
        name="Profile Timing",
        default=False,
        description="Show per-evaluation timing for Blender read/write glue and C++ deformation",
    )
    last_eval_timing: bpy.props.StringProperty(name="Last Eval Timing", default="", options={"SKIP_SAVE"})
    last_eval_blender_timing: bpy.props.StringProperty(name="Last Eval Blender Timing", default="", options={"SKIP_SAVE"})
    last_eval_gpu_timing: bpy.props.StringProperty(name="Last Eval GPU Timing", default="", options={"SKIP_SAVE"})
    last_train_timing: bpy.props.StringProperty(name="Last Train Timing", default="", options={"SKIP_SAVE"})
    trained: bpy.props.BoolProperty(name="Trained", default=False, options={"SKIP_SAVE"})
    status: bpy.props.StringProperty(name="Status", default="Not trained", options={"SKIP_SAVE"})


def get_settings(context: bpy.types.Context) -> PT_Settings:
    return context.scene.pose_trainer
