import bpy


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

    relax_iterations: bpy.props.IntProperty(name="Relax", default=10, min=1, max=50)
    solve_iterations: bpy.props.IntProperty(name="Solve", default=1, min=1, max=5)
    rbf_radius: bpy.props.FloatProperty(name="RBF Radius", default=0.1, min=0.0001)
    regularization: bpy.props.FloatProperty(name="Regularization", default=0.001, min=0.0)
    envelope: bpy.props.FloatProperty(name="Envelope", default=1.0, min=0.0, max=1.0)

    live_update: bpy.props.BoolProperty(name="Live Update", default=False)
    trained: bpy.props.BoolProperty(name="Trained", default=False, options={"SKIP_SAVE"})
    status: bpy.props.StringProperty(name="Status", default="Not trained", options={"SKIP_SAVE"})


def get_settings(context: bpy.types.Context) -> PT_Settings:
    return context.scene.pose_trainer
