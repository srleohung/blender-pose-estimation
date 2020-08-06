import bpy


class OBJECT_MT_PoseEstimationPanel(bpy.types.WorkSpaceTool):
    """Pose Estimation uses camera capture to estimate pose and change bone position"""
    bl_label = "Pose Estimation"
    bl_space_type = 'VIEW_3D'
    bl_context_mode='OBJECT'
    bl_idname = "ui_plus.pose_estimation"
    bl_options = {'REGISTER'}
    bl_icon = "ops.generic.select_circle"

    def draw_settings(context, layout, tool):
        row = layout.row()
        op = row.operator("wm.pose_estimation_operator", text="Capture", icon="OUTLINER_OB_CAMERA")

def register():
    bpy.utils.register_tool(OBJECT_MT_PoseEstimationPanel, separator=True, group=True)

def unregister():
    bpy.utils.unregister_tool(OBJECT_MT_PoseEstimationPanel)

if __name__ == "__main__":
    register()
