import sys
sys.path.append('/Users/leo/.local/lib/python3.7/site-packages')
sys.path.append('/Users/leo/blender-pose-estimation')

import bpy
from imutils import face_utils
import cv2
import time
import numpy
from bpy.props import FloatProperty
import math
# tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# posenet
import posenet
    
class PoseEstimationOperator(bpy.types.Operator):
    """Start capturing for pose estimation"""
    bl_idname = "wm.pose_estimation_operator"
    bl_label = "Pose Estimation Operator"  
    _timer = None
    _cap  = None 
    _id = "/Users/leo/blender-pose-estimation/testing.mp4"
    width = 800
    height = 600
    stop :bpy.props.BoolProperty()

    # posenet variable
    model = 101
    scale_factor = 0.7125
    
    # initialization tensorflow
    sess = None
    model_cfg = None
    model_outputs = None
    output_stride = None

    def modal(self, context, event):
        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_session()
            self.init_camera()
            _, image = self._cap.read()
            cv2.imshow('posenet', image)
            input_image, display_image, output_scale = posenet.read_cap(
                self._cap, scale_factor=self.scale_factor, output_stride=self.output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.sess.run(
                self.model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=self.output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            bones = bpy.data.objects["RIG-Vincent"].pose.bones
            pose = self.get_pose(keypoint_coords)
            
            if pose["nose"][0] == 0 and pose["nose"][1] == 0:
                return {'PASS_THROUGH'}  
            
            if not hasattr(self, 'first_pose'):
                    self.first_pose = pose
                    
            ##### ##### ##### ##### #####
            bones["head_fk"].rotation_euler[0] = self.smooth_value("head_fk", 3, (pose["nose"][1] - self.first_pose["nose"][1])) / 100   # Up/Down
            
            bones["shoulder_L"].location[0] = self.smooth_value("shoulder_L", 3, (pose["leftShoulder"][1] - self.first_pose["leftShoulder"][1])) / 1000 
            bones["shoulder_L"].location[1] = self.smooth_value("shoulder_L", 3, (pose["leftShoulder"][0] - self.first_pose["leftShoulder"][0])) / 1000   
            
            bones["shoulder_R"].location[0] = self.smooth_value("shoulder_R", 3, (pose["rightShoulder"][1] - self.first_pose["rightShoulder"][1])) / 1000  
            bones["shoulder_R"].location[1] = self.smooth_value("shoulder_R", 3, (pose["rightShoulder"][0] - self.first_pose["rightShoulder"][0])) / 1000  
            
            bones["hand_ik_ctrl_L"].location[2] = self.smooth_value("hand_ik_ctrl_L", 3, (pose["leftWrist"][1] - self.first_pose["leftWrist"][1])) / 1000  
            bones["hand_ik_ctrl_R"].location[2] = self.smooth_value("hand_ik_ctrl_R", 3, (pose["rightWrist"][1] - self.first_pose["rightWrist"][1])) / 1000  
            
            bones["sole_ctrl_L"].location[2] = self.smooth_value("sole_ctrl_L", 3, (pose["leftAnkle"][1] - self.first_pose["leftAnkle"][1])) / 1000  
            bones["sole_ctrl_R"].location[2] = self.smooth_value("sole_ctrl_R", 3, (pose["rightAnkle"][1] - self.first_pose["rightAnkle"][1])) / 1000              
            ##### ##### ##### ##### #####

            cv2.imshow('posenet', overlay_image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}   
    
    def get_pose(self, keypoint_coords):
        pose = {}
        pose["nose"] = keypoint_coords[0][0]
        pose["leftEye"] = keypoint_coords[0][1]
        pose["rightEye"] = keypoint_coords[0][2]
        pose["leftEar"] = keypoint_coords[0][3]
        pose["rightEar"] = keypoint_coords[0][4]
        pose["leftShoulder"] = keypoint_coords[0][5]
        pose["rightShoulder"] = keypoint_coords[0][6]
        pose["leftElbow"] = keypoint_coords[0][7]
        pose["rightElbow"] = keypoint_coords[0][8]
        pose["leftWrist"] = keypoint_coords[0][9]
        pose["rightWrist"] = keypoint_coords[0][10]
        pose["leftHip"] = keypoint_coords[0][11]
        pose["rightHip"] = keypoint_coords[0][12]
        pose["leftKnee"] = keypoint_coords[0][13]
        pose["rightKnee"] = keypoint_coords[0][14]
        pose["leftAnkle"] = keypoint_coords[0][15]
        pose["rightAnkle"] = keypoint_coords[0][16]
        return pose

    # Keeps a moving average of given length
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = numpy.array([value])
        else:
            self.smooth[name] = numpy.insert(arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = numpy.delete(self.smooth[name], self.smooth[name].size-1, 0)
        sum = 0
        for val in self.smooth[name]:
            sum += val
        return sum / self.smooth[name].size


    # Keeps min and max values, then returns the value in a ranve 0 - 1
    def get_range(self, name, value):
        if not hasattr(self, 'range'):
            self.range = {}
        if not name in self.range:
            self.range[name] = numpy.array([value, value])
        else:
            self.range[name] = numpy.array([min(value, self.range[name][0]), max(value, self.range[name][1])] )
        val_range = self.range[name][1] - self.range[name][0]
        if val_range != 0:
            return (value - self.range[name][0]) / val_range
        else:
            return 0

    def init_session(self):
        if self.sess == None:
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())
            self.model_cfg, self.model_outputs = posenet.load_model(self.model, self.sess)
            self.output_stride = self.model_cfg['output_stride']

    def init_camera(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture(self._id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.5)
    
    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        
    def execute(self, context):
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None
        self.sess.close()
        self.sess = None

def register():
    bpy.utils.register_class(PoseEstimationOperator)

def unregister():
    bpy.utils.unregister_class(PoseEstimationOperator)

if __name__ == "__main__":
    register()

