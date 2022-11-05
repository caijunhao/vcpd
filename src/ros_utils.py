from vcpd.srv import RequestVolume, RequestVolumeRequest, RequestVolumeResponse, RequestGripperPose, RequestGripperPoseResponse
from vcpd.msg import Volume, GripperPose
from franka_msgs.msg import ErrorRecoveryActionGoal
from cpn.utils import sample_contact_points, select_gripper_pose, clustering
from sdf import TSDF, PSDF, GradientSDF
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Pose
import quaternion
import numpy as np
import torch
import rospy
import threading
import time
import copy


def from_rs_intr_to_mat(rs_intrinsics):
    intrinsic = np.array([rs_intrinsics.fx, 0.0, rs_intrinsics.ppx,
                          0.0, rs_intrinsics.fy, rs_intrinsics.ppy,
                          0.0, 0.0, 1.0]).reshape(3, 3)
    return intrinsic


class Transform(object):
    def __init__(self, array):
        array = np.asanyarray(array)
        dtype = array.dtype
        if array.shape[0] == 7:  # px, py, pz, w, x, y, z
            self._pos_n_quat = array
            self._pose = np.eye(4, dtype=dtype)
            # https://github.com/moble/quaternion/blob/main/src/quaternion/__init__.py#L73
            self._pose[0:3, 0:3] = quaternion.as_rotation_matrix(quaternion.from_float_array(array[3:7]))
            self._pose[0:3, 3] = array[0:3]
        elif array.shape[0] == 4 and array.shape[1] == 4:
            self._pose = array
            quat = quaternion.as_float_array(quaternion.from_rotation_matrix(array[0:3, 0:3]))
            self._pos_n_quat = np.concatenate([array[0:3, 3], quat])
        else:
            raise ValueError('the array does not match the required shape, please check.')

    def __call__(self):
        return self._pose

    def rot(self):
        return self._pose[0:3, 0:3]

    def quat(self):
        return self._pos_n_quat[3:7]

    def pos(self):
        return self._pos_n_quat[0:3]


class SDFCommander(object):
    def __init__(self, sdf, pcl_pub):
        self.sdf = sdf
        self.pcl_pub = pcl_pub

        self.t_base2ee = None  # current pose from base frame to O_T_EE
        self.processed_flag = True  # flag to specify whether current visual data in this buffer is processed or not
        self.num_received = 0  # number of visual data received

        self.start_flag = False  # trigger flag to specify whether to receive data or not
        self.save_flag = False  # flag to save the mesh generated from sdf volume
        self.reset_flag = False  # flag to reset values for the sdf volume
        self.valid_flag = False  # flag to specify if it is valid to response to the request

        self.pose_lock = threading.Lock()
        self.sdf_lock = threading.Lock()

    def callback_t_base2ee(self, msg):
        ee_pose = np.asarray(msg.pose).reshape((4, 4), order='F').astype(np.float32)
        self.pose_lock.acquire()
        self.t_base2ee = Transform(ee_pose)
        self.pose_lock.release()

    def callback_enable(self, msg):
        """
        :param msg: vpn/Bool.msg
        """
        self.start_flag = msg.data
        if self.start_flag:
            rospy.loginfo("flag received, now start processing sdf")
        else:
            rospy.loginfo("flag received, pause processing")

    def callback_save(self, msg):
        """
        :param msg: vpn/Bool.msg
        """
        self.save_flag = msg.data
        rospy.loginfo("flag received, generate mesh from sdf and save to the disk")

    def callback_reset(self, msg):
        """
        :param msg: vpn/Bool.msg
        """
        self.reset_flag = msg.data
        rospy.loginfo("flag received, re-initialize sdf")

    def tsdf_integrate(self, depth, intrinsic, m_base2cam, rgb):
        self.sdf_lock.acquire()
        self.sdf.integrate(depth, intrinsic, m_base2cam, rgb=rgb)
        self.sdf_lock.release()

    def reset(self):
        self.sdf_lock.acquire()
        self.sdf.reset()
        self.reset_flag = False
        self.valid_flag = False
        self.sdf_lock.release()

    def save_mesh(self, name='out.ply'):
        if self.valid_flag:
            self.sdf_lock.acquire()
            v, f, n, rgb = self.sdf.marching_cubes(step_size=1)
            self.sdf_lock.release()
            self.sdf.write_mesh(name, v, f, n, rgb)
        else:
            rospy.logwarn('no visual data is merged into the volume currently, '
                          'no mesh will be saved at this moment.')
        self.save_flag = False

    def handle_send_tsdf_volume(self, req):
        while not self.valid_flag:
            rospy.sleep(0.01)
        self.sdf_lock.acquire()
        if req.post_processed:
            sdf_volume = self.sdf.post_processed_vol
        else:
            sdf_volume = self.sdf.sdf_vol
        self.sdf_lock.release()
        if req.gaussian_blur:
            sdf_volume = self.sdf.gaussian_smooth(sdf_volume)
        sdf_volume = sdf_volume.cpu().numpy()
        volume_msg = Volume()
        volume_msg.height, volume_msg.width, volume_msg.depth = sdf_volume.shape
        volume_msg.data = sdf_volume.reshape(-1).tolist()
        return RequestVolumeResponse(volume_msg)

    def callback_pcl(self, msg):
        while not self.valid_flag:
            rospy.loginfo('nothing in the volume ...')
            rospy.sleep(0.1)
            return
        if msg.data:
            # self.sdf_lock.acquire()
            xyz, rgb = self.sdf.compute_pcl(use_post_processed=True, smooth=True)
            rgb = rgb.to(xyz.dtype)
            xyz_rgb = torch.cat([xyz, rgb], dim=1)
            # self.sdf_lock.release()
            xyz_rgb = xyz_rgb.cpu().numpy()
            pcl2msg = self.array_to_pointcloud2(xyz_rgb)
            self.pcl_pub.publish(pcl2msg)
        rospy.sleep(0.1)

    @property
    def m_base2ee(self):
        self.pose_lock.acquire()
        m_base2ee = copy.deepcopy(self.t_base2ee())  # current pose from ee to base
        self.pose_lock.release()
        return m_base2ee

    @staticmethod
    def array_to_pointcloud2(pcl):
        # https://blog.csdn.net/huyaoyu/article/details/103193523
        # https://answers.ros.org/question/195962/rviz-fixed-frame-world-does-not-exist/
        # rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map 10
        if not pcl.dtype == np.float32:
            pcl = pcl.astype(np.float32)
        pcl = np.atleast_2d(pcl)  # make it 2d (even if height will be 1)
        num_pts = pcl.shape[0]
        xyz = pcl[:, 0:3]
        rgba = np.ones((num_pts, 4), dtype=np.uint8) * 255
        rgba[:, 0:3] = pcl[:, 3:6].astype(np.uint8)
        rgba = rgba.view('uint32')
        pcl = np.zeros((num_pts, 1), dtype={'names': ('x', 'y', 'z', 'rgba'),
                                            'formats': ('f4', 'f4', 'f4', 'u4')})
        pcl['x'] = xyz[:, 0].reshape(-1, 1)
        pcl['y'] = xyz[:, 1].reshape(-1, 1)
        pcl['z'] = xyz[:, 2].reshape(-1, 1)
        pcl['rgba'] = rgba
        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = 'map'
        msg.height = 1
        msg.width = pcl.shape[0]
        msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1),
                      PointField('rgba', 12, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * pcl.shape[0]
        msg.is_dense = False  # int(np.isfinite(xyz).all())
        msg.data = pcl.tostring()
        return msg


class CPNCommander(object):
    def __init__(self, volume_bounds, voxel_length, model, gripper_points, gripper_depth, device):
        self.cpn = model
        resolution = np.ceil((volume_bounds[1] - volume_bounds[0] - voxel_length / 7) / voxel_length)
        self.sdf = TSDF(volume_bounds[0], resolution, voxel_length, device=device)
        self.device = device
        self.trigger_flag = False
        self.gpr_pts = gripper_points  # Nx3-d torch tensor
        self.gpr_d = gripper_depth  # the approach distance between the end point of the finger and the gripper frame

    def callback_cpn(self, msg):
        self.trigger_flag = msg.data
        if self.trigger_flag:
            rospy.loginfo('activate cpn node')
        else:
            rospy.loginfo('pause cpn node')

    def request_a_volume(self, post_processed=True, gaussian_blur=True, service_name='request_a_volume'):
        rospy.wait_for_service(service_name)
        try:
            handle_receive_sdf_volume = rospy.ServiceProxy(service_name,
                                                           RequestVolume)
            req = RequestVolumeRequest()
            req.post_processed = post_processed
            req.gaussian_blur = gaussian_blur
            res = handle_receive_sdf_volume(req)
            rospy.loginfo('receive a volume message from {}'.format(service_name))
            return self.volume_msg2numpy(res.v)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def inference(self):
        sdf_volume = self.request_a_volume()
        self.sdf.sdf_vol = torch.from_numpy(sdf_volume).to(self.device)
        try:
            cp1, cp2 = sample_contact_points(self.sdf, post_processed=False, gaussian_blur=False, step_size=1)
        except ValueError:
            rospy.logwarn('cannot find level set from the volume, try next one')
            return None
        if cp1.shape[0] == 0:
            rospy.logwarn('cannot find potential contact points')
            return None
        ids_cp1, ids_cp2 = self.sdf.get_ids(cp1), self.sdf.get_ids(cp2)
        sample = dict()
        sample['sdf_volume'] = self.sdf.sdf_vol.unsqueeze(dim=0).unsqueeze(dim=0)
        sample['ids_cp1'] = ids_cp1.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
        sample['ids_cp2'] = ids_cp2.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
        out = torch.squeeze(self.cpn.forward(sample))
        if out.shape[0] == 0:
            rospy.logwarn('output size is zero')
            return None
        gripper_pos, rot, width, cp1, cp2 = select_gripper_pose(self.sdf, self.gpr_pts, out, cp1, cp2, self.gpr_d,
                                                                check_tray=False, post_processed=False,
                                                                gaussian_blur=False)
        gripper_pos, rot, width, cp1, cp2 = clustering(gripper_pos, rot, width, cp1, cp2)
        msg = self.pose2msg(cp1, cp2, rot, width)
        return msg

    @staticmethod
    def pose2msg(cp1, cp2, rot, width):
        msg = GripperPose()
        msg.cp1.x, msg.cp1.y, msg.cp1.z = cp1.tolist()
        msg.cp2.x, msg.cp2.y, msg.cp2.z = cp2.tolist()
        msg.av.x, msg.av.y, msg.av.z = rot[:, 2].tolist()
        msg.gv.x, msg.gv.y, msg.gv.z = rot[:, 1].tolist()
        msg.width = width
        return msg

    @staticmethod
    def volume_msg2numpy(msg):
        volume = np.array(msg.data).reshape((msg.height, msg.width, msg.depth))
        return volume.astype(np.float32)


class PandaGripperPoseParser(object):
    def __init__(self, topic_name):
        _ = rospy.Subscriber(topic_name, GripperPose, self.callback)
        s = rospy.Service('request_panda_gripper_pose', RequestGripperPose, self.handle_send_gripper_pose)
        self._pos, self._quat, self._width = None, None, 0.08
        self._lock = threading.Lock()

    def callback(self, msg):
        pos, quat, width = self.msg2pose(msg)
        self._lock.acquire()
        self._pos, self._quat, self._width = pos, quat, width
        self._lock.release()

    def get(self):
        self._lock.acquire()
        pos, quat, width = copy.deepcopy(self._pos), copy.deepcopy(self._quat), self._width
        self._lock.release()
        return pos, quat, width

    def reset(self):
        self._lock.acquire()
        self._pos, self._quat, self._width = None, None, 0.08
        self._lock.release()

    def handle_send_gripper_pose(self, req):
        msg = Pose()
        if self._pos is None:
            msg.position.x, msg.position.y, msg.position.z = 0, 0, 0
            msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = 1, 0, 0, 0
        else:
            msg.position.x, msg.position.y, msg.position.z = self._pos.tolist()
            msg.orientation.w = self._quat.w
            msg.orientation.x = self._quat.x
            msg.orientation.y = self._quat.y
            msg.orientation.z = self._quat.z
        return RequestGripperPoseResponse(msg)

    @staticmethod
    def msg2pose(msg):
        """
        Convert GripperPose message to ee pose of panda
        :param msg: GripperPose.msg
        :return: A 4x4-d numpy array representing the pose of the end effector.
        """
        cp1 = np.array([msg.cp1.x, msg.cp1.y, msg.cp1.z])
        cp2 = np.array([msg.cp2.x, msg.cp2.y, msg.cp2.z])
        pos = (cp1 + cp2) / 2
        # for panda robot, the grasp vector lies on x-axis
        z = np.array([msg.av.x, msg.av.y, msg.av.z])
        y = np.array([msg.gv.x, msg.gv.y, msg.gv.z])
        x = np.cross(y, z)
        x = x if x[0] > 0 else -x
        y = np.cross(z, x)
        rot = np.stack([x, y, z], axis=1)
        quat = quaternion.from_rotation_matrix(rot)
        return pos, quat, msg.width


class PandaCommander(object):
    def __init__(self, robot, arm_speed=0.05, rate=100):
        self.robot = robot
        # self.robot.set_arm_speed(arm_speed)
        self.gripper = robot.get_gripper()
        self.lock = threading.Lock()
        self.v_limits = np.array(self.robot.get_joint_limits().velocity)
        self.err_rec_pub = rospy.Publisher('/franka_ros_interface/franka_control/error_recovery/goal',
                                           ErrorRecoveryActionGoal,
                                           queue_size=1)
        self.rate = rate
        self.arm_speed = arm_speed
        self.zero_vel = np.zeros(7, dtype=np.float64).tolist()
        self.error = False

    def detect_grasp(self):
        pos = self.gripper.joint_positions()
        return pos['panda_finger_joint1'] > 0.001 and pos['panda_finger_joint2'] > 0.001

    def set_gripper_width(self, width, wait=False):
        width = max(0.0, min(0.08, width))
        self.gripper.move_joints(width, wait_for_result=wait)

    def open_gripper(self):
        self.gripper.grasp(0.085, 0, 0.05)

    def close_gripper(self):
        self.set_gripper_width(0.0)

    def ee_pose(self):
        return self.robot.ee_pose()

    def angles(self):
        return self.robot.angles()

    def jnt_vals(self):
        return self.robot.angles()

    def movej(self, jnt_vals, unit='r'):
        jnt_vals = np.asarray(jnt_vals)
        if unit == 'd':
            jnt_vals = np.deg2rad(jnt_vals)
        self.robot.move_to_joint_position(jnt_vals)

    def movel(self, pos, quat):
        self.robot.move_to_cartesian_pose(pos, quat)

    def fk(self, jnt_vals, unit='r'):
        jnt_vals = np.asarray(jnt_vals)
        if unit == 'd':
            jnt_vals = np.deg2rad(jnt_vals)
        pos, quat = self.robot.forward_kinematics(jnt_vals)
        return pos, quat

    def ik(self, pos, quat):
        jnt_vals = self.robot.inverse_kinematics(pos, quat)
        return jnt_vals

    def zeroize_vel(self, th=1e-3):
        vel_norm = np.linalg.norm(np.array(list(self.robot.joint_velocities().values())[:]))
        i = 0
        while vel_norm > th:
            self.robot.exec_velocity_cmd(self.zero_vel)
            rospy.sleep(1 / self.rate)
            vel_norm = np.linalg.norm(np.array(list(self.robot.joint_velocities().values())[:]))
            i += 1
        rospy.loginfo('stop joint velocity controller with {} zero vel cmd(s)'.format(i))

    def vel_ctl(self, pos, quat, min_height=None, ori_ctl=False, gain_h=8):
        b = time.time()
        if pos is None:
            rospy.loginfo('no valid pose detected')
            self.robot.exec_velocity_cmd(self.zero_vel)
            rospy.sleep(1 / self.rate)
            return
        rospy.loginfo('target position of T_base2ee: {}'.format(pos))
        curr_pos, curr_quat = self.robot.ee_pose()
        pos_res = pos - curr_pos
        norm = np.linalg.norm(pos_res)
        direction = pos_res / norm
        quat_res = quaternion.as_float_array(quat * curr_quat.conj())
        ori_res = np.sign(quat_res[0]) * quat_res[1:]
        # since we stop at the desired height rather than the target pose,
        # we use signed gain here to let end-effector keep stable at desired height location
        if min_height is None:
            pos_gain = min(norm, self.arm_speed)
            pos_gain_vec = pos_gain
        else:
            pos_gain = curr_pos[2] - min_height
            pos_gain = min(np.abs(pos_gain), self.arm_speed)  # truncated gain ?? np.sign(pos_gain) *
            pos_gain_vec = np.ones(3) * pos_gain
            pos_gain_vec[0:2] *= gain_h  # amplify gain in x and y direction
        ori_gain = 1 if ori_ctl else 0
        next_vel = np.concatenate([pos_gain_vec * direction, ori_gain * ori_res], axis=0)
        jacobian = self.robot.jacobian()
        jacobian_pseudo_inverse = np.linalg.pinv(jacobian)
        next_jnt_vel = np.dot(jacobian_pseudo_inverse, next_vel)
        e = time.time()
        rospy.loginfo('estimated rate for computing vel cmd: {}Hz'.format((1/(e-b))))
        self.robot.exec_velocity_cmd(next_jnt_vel)
        rospy.sleep(max(1/self.rate-(e-b), 0))

    # https://github.com/justagist/franka_ros_interface/blob/master/franka_common/franka_core_msgs/msg/RobotState.msg
    def robot_state_callback(self, msg):
        for name in dir(msg.current_errors):
            if name[0] != '_' and 'serialize' not in name:
                if getattr(msg.current_errors, name):
                    self.recover_from_errors()
                    return
        if msg.robot_mode == 4:
            self.recover_from_errors()
            return
        rospy.sleep(0.01)

    def recover_from_errors(self):
        rospy.loginfo('call the error reset action server')
        self.error = True
        self.err_rec_pub.publish(ErrorRecoveryActionGoal())
        rospy.sleep(3.0)
        self.error = False
