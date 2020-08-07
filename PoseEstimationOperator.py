import bpy
from imutils import face_utils
import dlib
import cv2
import time
import numpy
from bpy.props import FloatProperty
import math
# tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# posenet
import sys
sys.path.append('/Users/leohung/blender-pose-estimation')
import posenet
    
class PoseEstimationOperator(bpy.types.Operator):
    """Start capturing for pose estimation"""
    bl_idname = "wm.pose_estimation_operator"
    bl_label = "Pose Estimation Operator"  
    _timer = None
    _cap  = None 
    _id = 0 # "/Users/leohung/Downloads/testing.mp4"
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

    # initialization calibrator
    proportion = None
    object_shoulder_width = 1.9
    calibration_times = 10
    calibration_count = 0
    original_nose = None
    original_leftEye = None
    original_rightEye = None
    original_leftEar = None
    original_rightEar = None
    original_leftShoulder = None
    original_rightShoulder = None
    original_leftElbow = None
    original_rightElbow = None
    original_leftWrist = None
    original_rightWrist = None
    original_leftHip = None
    original_rightHip = None
    original_leftKnee = None
    original_rightKnee = None
    original_leftAnkle = None
    original_rightAnkle = None
    keyframe_insert_enable = False

    def modal(self, context, event):
        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_session()
            self.init_camera()
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

            bones = bpy.data.objects["rig"].pose.bones

            if self.calibration_count < self.calibration_times:
                self.original_nose = keypoint_coords[0][0]
                self.original_leftEye = keypoint_coords[0][1]
                self.original_rightEye = keypoint_coords[0][2]
                self.original_leftEar = keypoint_coords[0][3]
                self.original_rightEar = keypoint_coords[0][4]
                self.original_leftShoulder = keypoint_coords[0][5]
                self.original_rightShoulder = keypoint_coords[0][6]
                self.original_leftElbow = keypoint_coords[0][7]
                self.original_rightElbow = keypoint_coords[0][8]
                self.original_leftWrist = keypoint_coords[0][9]
                self.original_rightWrist = keypoint_coords[0][10]
                self.original_leftHip = keypoint_coords[0][11]
                self.original_rightHip = keypoint_coords[0][12]
                self.original_leftKnee = keypoint_coords[0][13]
                self.original_rightKnee = keypoint_coords[0][14]
                self.original_leftAnkle = keypoint_coords[0][15]
                self.original_rightAnkle = keypoint_coords[0][16]
                self.proportion = self.object_shoulder_width / math.sqrt(math.pow(self.original_leftShoulder[0] - self.original_rightShoulder[0], 2) + math.pow(self.original_leftShoulder[1] - self.original_rightShoulder[1], 2))
                self.calibration_count += 1
            else:
                # Move nose
                nose = keypoint_coords[0][0]
                nose = [(self.original_nose[0] - nose[0])*self.proportion, (self.original_nose[1] - nose[1])*self.proportion]
                bones["head"].rotation_mode = 'XYZ'
                bones["head"].rotation_euler[0] = nose[0] * -1
                bones["head"].rotation_euler[1] = nose[1]
                
                # Move eyes
                leftEye = keypoint_coords[0][1]
                leftEye = [(self.original_leftEye[0] - leftEye[0])*self.proportion, (self.original_leftEye[1] - leftEye[1])*self.proportion]

                rightEye = keypoint_coords[0][2]
                rightEye = [(self.original_rightEye[0] - rightEye[0])*self.proportion, (self.original_rightEye[1] - rightEye[1])*self.proportion]
                
                # Move ears
                leftEar = keypoint_coords[0][3]
                leftEar = [(self.original_leftEar[0] - leftEar[0])*self.proportion, (self.original_leftEar[1] - leftEar[1])*self.proportion]
                
                rightEar = keypoint_coords[0][4]
                rightEar = [(self.original_rightEar[0] - rightEar[0])*self.proportion, (self.original_rightEar[1] - rightEar[1])*self.proportion]
                
                # Move shoulders
                leftShoulder = keypoint_coords[0][5]
                leftShoulder = [(self.original_leftShoulder[0] - leftShoulder[0])*self.proportion, (self.original_leftShoulder[1] - leftShoulder[1])*self.proportion]

                rightShoulder = keypoint_coords[0][6]
                rightShoulder = [(self.original_rightShoulder[0] - rightShoulder[0])*self.proportion, (self.original_rightShoulder[1] - rightShoulder[1])*self.proportion]
                bones["chest"].location[2] = (leftShoulder[0] + rightShoulder[0]) / 2
                bones["chest"].location[0] = (leftShoulder[1] + rightShoulder[1]) / 2

                # Move elbows
                leftElbow = keypoint_coords[0][7]
                leftElbow = [(self.original_leftElbow[0] - leftElbow[0])*self.proportion, (self.original_leftElbow[1] - leftElbow[1])*self.proportion]
                
                rightElbow = keypoint_coords[0][8]
                rightElbow = [(self.original_rightElbow[0] - rightElbow[0])*self.proportion, (self.original_rightElbow[1] - rightElbow[1])*self.proportion]
                
                # Move wrists
                leftWrist = keypoint_coords[0][9]
                leftWrist = [(self.original_leftWrist[0] - leftWrist[0])*self.proportion, (self.original_leftWrist[1] - leftWrist[1])*self.proportion]
                bones["hand_ik.L"].location[2] = leftWrist[0]
                bones["hand_ik.L"].location[1] = leftWrist[1] * -1
                
                rightWrist = keypoint_coords[0][10]
                rightWrist = [(self.original_rightWrist[0] - rightWrist[0])*self.proportion, (self.original_rightWrist[1] - rightWrist[1])*self.proportion]
                bones["hand_ik.R"].location[2] = rightWrist[0]
                bones["hand_ik.R"].location[1] = rightWrist[1]
                
                # Move hips
                leftHip = keypoint_coords[0][11]
                leftHip = [(self.original_leftHip[0] - leftHip[0])*self.proportion, (self.original_leftHip[1] - leftHip[1])*self.proportion]
                
                rightHip = keypoint_coords[0][12]
                rightHip = [(self.original_rightHip[0] - rightHip[0])*self.proportion, (self.original_rightHip[1] - rightHip[1])*self.proportion]
                bones["torso"].location[2] = (leftHip[0] + rightHip[0]) / 2
                bones["torso"].location[0] = (leftHip[1] + rightHip[1]) / 2

                # Move knees
                leftKnee = keypoint_coords[0][13]
                leftKnee = [(self.original_leftKnee[0] - leftKnee[0])*self.proportion, (self.original_leftKnee[1] - leftKnee[1])*self.proportion]
                
                rightKnee = keypoint_coords[0][14]
                rightKnee = [(self.original_rightKnee[0] - rightKnee[0])*self.proportion, (self.original_rightKnee[1] - rightKnee[1])*self.proportion]
                
                # Move ankles
                leftAnkle = keypoint_coords[0][15]
                leftAnkle = [(self.original_leftAnkle[0] - leftAnkle[0])*self.proportion, (self.original_leftAnkle[1] - leftAnkle[1])*self.proportion]
                bones["foot_ik.L"].location[2] = leftAnkle[0]
                bones["foot_ik.L"].location[0] = leftAnkle[1]

                rightAnkle = keypoint_coords[0][16]
                rightAnkle = [(self.original_rightAnkle[0] - rightAnkle[0])*self.proportion, (self.original_rightAnkle[1] - rightAnkle[1])*self.proportion]
                bones["foot_ik.R"].location[2] = rightAnkle[0]
                bones["foot_ik.R"].location[0] = rightAnkle[1]

                if self.keyframe_insert_enable == True:
                    bones["head"].keyframe_insert(data_path="rotation_euler", index=-1)
                    bones["chest"].keyframe_insert(data_path="location", index=-1)
                    bones["hand_ik.L"].keyframe_insert(data_path="location", index=-1)
                    bones["hand_ik.R"].keyframe_insert(data_path="location", index=-1)
                    bones["torso"].keyframe_insert(data_path="location", index=-1)
                    bones["foot_ik.L"].keyframe_insert(data_path="location", index=-1)
                    bones["foot_ik.R"].keyframe_insert(data_path="location", index=-1)

            cv2.imshow('posenet', overlay_image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}
    
    def init_session(self):
        if self.sess == None:
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())
            self.model_cfg, self.model_outputs = posenet.load_model(self.model, self.sess)
            self.output_stride = self.model_cfg['output_stride']
            self.calibration_count = 0
            self.proportion = None

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
