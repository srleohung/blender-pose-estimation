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
import sys
sys.path.append('/Users/leo/.local/lib/python3.7/site-packages')

sys.path.append('/Users/leo/blender-pose-estimation')
import posenet
    
class PoseEstimationOperator(bpy.types.Operator):
    """Start capturing for pose estimation"""
    bl_idname = "wm.pose_estimation_operator"
    bl_label = "Pose Estimation Operator"  
    _timer = None
    _cap  = None 
    # _id = "/Users/leohung/Downloads/testing.mp4"
    _id = 0
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
    object_shoulder_width = 2.3 # 2.3
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
                if self.original_leftShoulder[0] == 0 or self.original_leftShoulder[1] == 0 or self.original_rightShoulder[0] == 0 or self.original_rightShoulder[1] == 0:
                    return {'PASS_THROUGH'}
                self.proportion = self.object_shoulder_width / math.sqrt(math.pow(self.original_leftShoulder[0] - self.original_rightShoulder[0], 2) + math.pow(self.original_leftShoulder[1] - self.original_rightShoulder[1], 2))
                self.calibration_count += 1
            else:
                x, y = self.calibration_location(keypoint_coords)
                # bones["root"].location[2] = y*self.proportion
                # bones["root"].location[0] = x*self.proportion

                # Rotate nose
                nose = keypoint_coords[0][0]
                nose = [(self.original_nose[0] - nose[0])*self.proportion, (self.original_nose[1] - nose[1])*self.proportion]
                bones["head"].rotation_mode = 'XYZ'
                bones["head"].rotation_euler[0] = self.smooth_value("head.x", 3, self.calibration_value(nose[0] * -1, 1, -1))
                bones["head"].rotation_euler[1] = self.smooth_value("head.y", 3, self.calibration_value(nose[1], 1, -1))
                
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

                # Move elbows
                leftElbow = keypoint_coords[0][7]
                leftElbow = [(self.original_leftElbow[0] - leftElbow[0])*self.proportion, (self.original_leftElbow[1] - leftElbow[1])*self.proportion*-1]
                x_x = 0.219534
                x_y = 0.650383
                x_z = 0.72719
                z_x = 0.17474
                z_y = -0.759534
                z_z = 0.626558
                bones["forearm_tweak.L"].location[0] = self.smooth_value("forearm_tweak.L.x", 2, leftElbow[1] * x_x + leftElbow[0] * z_x)
                bones["forearm_tweak.L"].location[1] = self.smooth_value("forearm_tweak.L.y", 2, leftElbow[1] * x_y + leftElbow[0] * z_y)
                bones["forearm_tweak.L"].location[2] = self.smooth_value("forearm_tweak.L.z", 2, leftElbow[1] * x_z + leftElbow[0] * z_z)
                # bones["forearm_tweak.L"].location[2] = self.smooth_value("forearm_tweak.L.y", 3, leftElbow[1])
                # bones["forearm_tweak.L"].location[0] = self.smooth_value("forearm_tweak.L.x", 3, leftElbow[0])

                rightElbow = keypoint_coords[0][8]
                rightElbow = [(self.original_rightElbow[0] - rightElbow[0])*self.proportion, (self.original_rightElbow[1] - rightElbow[1])*self.proportion*-1]
                x_x = 0.219243
                x_y = -0.650383
                x_z = -0.727278
                z_x = -0.174489
                z_y = -0.759534
                z_z = 0.626628
                bones["forearm_tweak.R"].location[0] = self.smooth_value("forearm_tweak.R.x", 2, rightElbow[1] * x_x + rightElbow[0] * z_x)
                bones["forearm_tweak.R"].location[1] = self.smooth_value("forearm_tweak.R.y", 2, rightElbow[1] * x_y + rightElbow[0] * z_y)
                bones["forearm_tweak.R"].location[2] = self.smooth_value("forearm_tweak.R.z", 2, rightElbow[1] * x_z + rightElbow[0] * z_z)
                # bones["forearm_tweak.R"].location[2] = self.smooth_value("forearm_tweak.R.y", 3, rightElbow[1])
                # bones["forearm_tweak.R"].location[0] = self.smooth_value("forearm_tweak.R.x", 3, rightElbow[0])

                # Move wrists
                leftWrist = keypoint_coords[0][9]
                leftWrist = [(self.original_leftWrist[0] - leftWrist[0])*self.proportion, (self.original_leftWrist[1] - leftWrist[1])*self.proportion*-1]
                print(leftWrist)
                x_x = 0.222131
                x_y = 0.667043
                x_z = 0.711134
                z_x = 0.171844
                z_y = -0.744722
                z_z = 0.644871
                bones["hand_ik.L"].location[0] = self.smooth_value("hand_ik.L.x", 3, leftWrist[1] * x_x + leftWrist[0] * z_x)
                bones["hand_ik.L"].location[1] = self.smooth_value("hand_ik.L.y", 3, leftWrist[1] * x_y + leftWrist[0] * z_y)
                bones["hand_ik.L"].location[2] = self.smooth_value("hand_ik.L.z", 3, leftWrist[1] * x_z + leftWrist[0] * z_z)
                # bones["hand_ik.L"].location[2] = self.smooth_value("hand_ik.L.y", 3, leftWrist[1])
                # bones["hand_ik.L"].location[0] = self.smooth_value("hand_ik.L.x", 3, leftWrist[0])

                rightWrist = keypoint_coords[0][10]
                rightWrist = [(self.original_rightWrist[0] - rightWrist[0])*self.proportion, (self.original_rightWrist[1] - rightWrist[1])*self.proportion*-1]
                x_x = 0.118835
                x_y = -0.801089
                x_z = -0.586629
                z_x = -0.251902
                z_y = -0.59581
                z_z = 0.762598
                bones["hand_ik.R"].location[0] = self.smooth_value("hand_ik.R.x", 3, rightWrist[1] * x_x + rightWrist[0] * z_x)
                bones["hand_ik.R"].location[1] = self.smooth_value("hand_ik.R.y", 3, rightWrist[1] * x_y + rightWrist[0] * z_y)
                bones["hand_ik.R"].location[2] = self.smooth_value("hand_ik.R.z", 3, rightWrist[1] * x_z + rightWrist[0] * z_z)
                # bones["hand_ik.R"].location[2] = self.smooth_value("hand_ik.R.y", 3, rightWrist[1])
                # bones["hand_ik.R"].location[0] = self.smooth_value("hand_ik.R.x", 3, rightWrist[0])

                # Rotate hand
                bones["hand_ik.L"].rotation_mode = 'XYZ'
                bones["hand_ik.L"].rotation_euler[0] = self.smooth_value("hand_ik.L.r", 3, math.tan((keypoint_coords[0][7][0] - keypoint_coords[0][9][0]) / math.sqrt(math.pow(keypoint_coords[0][7][0] - keypoint_coords[0][9][0], 2) + math.pow(keypoint_coords[0][7][1] - keypoint_coords[0][9][1], 2)))) + 1.2
                bones["hand_ik.R"].rotation_mode = 'XYZ'
                bones["hand_ik.R"].rotation_euler[0] = self.smooth_value("hand_ik.R.r", 3, math.tan((keypoint_coords[0][8][0] - keypoint_coords[0][10][0]) / math.sqrt(math.pow(keypoint_coords[0][8][0] - keypoint_coords[0][10][0], 2) + math.pow(keypoint_coords[0][8][1] - keypoint_coords[0][10][1], 2)))) + 1.2
                
                # Move hips
                leftHip = keypoint_coords[0][11]
                leftHip = [(self.original_leftHip[0] - leftHip[0])*self.proportion, (self.original_leftHip[1] - leftHip[1])*self.proportion]

                rightHip = keypoint_coords[0][12]
                rightHip = [(self.original_rightHip[0] - rightHip[0])*self.proportion, (self.original_rightHip[1] - rightHip[1])*self.proportion]
                bones["torso"].location[2] = self.smooth_value("torso.y", 2, (leftHip[1] + rightHip[1]) / 2)
                bones["torso"].location[0] = self.smooth_value("torso.x", 2, (leftHip[0] + rightHip[0]) / 2)

                # Move knees
                leftKnee = keypoint_coords[0][13]
                leftKnee = [(self.original_leftKnee[0] - leftKnee[0])*self.proportion, (self.original_leftKnee[1] - leftKnee[1])*self.proportion]
                # bones["shin_tweak.L"].location[2] = self.smooth_value("torso.y", 2, leftKnee[1])
                # bones["shin_tweak.L"].location[0] = self.smooth_value("torso.x", 2, leftKnee[0])

                rightKnee = keypoint_coords[0][14]
                rightKnee = [(self.original_rightKnee[0] - rightKnee[0])*self.proportion, (self.original_rightKnee[1] - rightKnee[1])*self.proportion]
                # bones["shin_tweak.R"].location[2] = rightKnee[0]
                # bones["shin_tweak.R"].location[1] = rightKnee[1]

                # Move ankles
                leftAnkle = keypoint_coords[0][15]
                leftAnkle = [(self.original_leftAnkle[0] - leftAnkle[0])*self.proportion, (self.original_leftAnkle[1] - leftAnkle[1])*self.proportion]
                bones["foot_ik.L"].location[2] = self.smooth_value("foot_ik.L.y", 2, leftAnkle[1])
                bones["foot_ik.L"].location[0] = self.smooth_value("foot_ik.L.x", 2, leftAnkle[0])

                rightAnkle = keypoint_coords[0][16]
                rightAnkle = [(self.original_rightAnkle[0] - rightAnkle[0])*self.proportion, (self.original_rightAnkle[1] - rightAnkle[1])*self.proportion]
                bones["foot_ik.R"].location[2] = self.smooth_value("foot_ik.R.y", 2, rightAnkle[1])
                bones["foot_ik.R"].location[0] = self.smooth_value("foot_ik.R.x", 2, rightAnkle[0])

                if self.keyframe_insert_enable == True:
                    # rotation_euler
                    bones["head"].keyframe_insert(data_path="rotation_euler", index=-1)
                    # location
                    bones["root"].keyframe_insert(data_path="location", index=-1)
                    bones["chest"].keyframe_insert(data_path="location", index=-1)
                    bones["forearm_tweak.L"].keyframe_insert(data_path="location", index=-1)
                    bones["forearm_tweak.R"].keyframe_insert(data_path="location", index=-1)
                    bones["hand_ik.L"].keyframe_insert(data_path="location", index=-1)
                    bones["hand_ik.R"].keyframe_insert(data_path="location", index=-1)
                    bones["torso"].keyframe_insert(data_path="location", index=-1)
                    bones["shin_tweak.L"].keyframe_insert(data_path="location", index=-1)
                    bones["shin_tweak.R"].keyframe_insert(data_path="location", index=-1)
                    bones["foot_ik.L"].keyframe_insert(data_path="location", index=-1)
                    bones["foot_ik.R"].keyframe_insert(data_path="location", index=-1)

            cv2.imshow('posenet', overlay_image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}   

    def calibration_value(self, value, _max, _min):
        if value > _max:
            return _max
        elif value < _min:
            return _min
        else:
            return value
    
    def calibration_location(self, keypoint_coords):
        x = (self.original_leftShoulder[0] + self.original_rightShoulder[0]) / 2 - (keypoint_coords[0][5][0] + keypoint_coords[0][6][0]) / 2
        y = (self.original_leftShoulder[1] + self.original_rightShoulder[1]) / 2 - (keypoint_coords[0][5][1] + keypoint_coords[0][6][1]) / 2
        self.original_nose = [self.original_nose[0] - x, self.original_nose[1] - y]
        self.original_leftEye = [self.original_leftEye[0] - x, self.original_leftEye[1] - y]
        self.original_rightEye = [self.original_rightEye[0] - x, self.original_rightEye[1] - y]
        self.original_leftEar = [self.original_leftEar[0] - x, self.original_leftEar[1] - y]
        self.original_rightEar = [self.original_rightEar[0] - x, self.original_rightEar[1] - y]
        self.original_leftShoulder = [self.original_leftShoulder[0] - x, self.original_leftShoulder[1] - y]
        self.original_rightShoulder = [self.original_rightShoulder[0] - x, self.original_rightShoulder[1] - y]
        self.original_leftElbow = [self.original_leftElbow[0] - x, self.original_leftElbow[1] - y]
        self.original_rightElbow = [self.original_rightElbow[0] - x, self.original_rightElbow[1] - y]
        self.original_leftWrist = [self.original_leftWrist[0] - x, self.original_leftWrist[1] - y]
        self.original_rightWrist = [self.original_rightWrist[0] - x, self.original_rightWrist[1] - y]
        self.original_leftHip = [self.original_leftHip[0] - x, self.original_leftHip[1] - y]
        self.original_rightHip = [self.original_rightHip[0] - x, self.original_rightHip[1] - y]
        self.original_leftKnee = [self.original_leftKnee[0] - x, self.original_leftKnee[1] - y]
        self.original_rightKnee = [self.original_rightKnee[0] - x, self.original_rightKnee[1] - y]
        self.original_leftAnkle = [self.original_leftAnkle[0] - x, self.original_leftAnkle[1] - y]
        self.original_rightAnkle = [self.original_rightAnkle[0] - x, self.original_rightAnkle[1] - y]
        self.proportion = self.object_shoulder_width / math.sqrt(math.pow(self.original_leftShoulder[0] - self.original_rightShoulder[0], 2) + math.pow(self.original_leftShoulder[1] - self.original_rightShoulder[1], 2))
        return x, y

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
