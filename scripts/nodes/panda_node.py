#!/usr/bin/env python3
from franka_core_msgs.msg import RobotState
from std_msgs.msg import Bool
from ros_utils import PandaCommander, PandaGripperPoseParser
from panda_robot import PandaArm
import numpy as np
import quaternion
import rospy
cfg = rospy.get_param


def panda_node():
    rospy.init_node('panda_node', anonymous=True)
    r = PandaArm()
    pc = PandaCommander(r, arm_speed=cfg('arm_speed'), rate=cfg('vel_rate'))
    _ = rospy.Subscriber(cfg('robot_state'), RobotState, pc.robot_state_callback)
    center = np.array([cfg('c_x'), cfg('c_y_aligned'), cfg('c_z')])
    bottom = center.copy()
    bottom[2] -= cfg('sdf_d') / 2 * cfg('voxel_length')
    pgpp = PandaGripperPoseParser('/cpn/gripper_pose')
    pc.movej(cfg('jnt_vals'))
    pc.gripper.grasp(0.085, 0, 0.05)
    drop_pos = [-1.5427169, -0.70545206, 1.95252443, -1.28864723, 0.64465563, 1.59058326, 1.28369785]
    tsdf_enable_pub = rospy.Publisher('/sdf/enable', Bool, queue_size=1)
    tsdf_reset_pub = rospy.Publisher('/sdf/reset', Bool, queue_size=1)
    tsdf_save_pub = rospy.Publisher('/sdf/save', Bool, queue_size=1)
    cpn_pub = rospy.Publisher('/cpn/flag', Bool, queue_size=1)
    true, false = Bool(True), Bool(False)
    rospy.sleep(0.5)
    while not rospy.is_shutdown():
        pgpp.reset()
        tsdf_reset_pub.publish(true)
        tsdf_enable_pub.publish(true)
        cpn_pub.publish(true)
        diff_height = np.abs(cfg('min_height') - pc.ee_pose()[0][2])
        while diff_height > 5e-3:
            pos, _, _ = pgpp.get()
            curr_pos = pc.ee_pose()[0]
            z = bottom - curr_pos
            z = z / np.linalg.norm(z)
            x = np.zeros_like(z)
            z_norm = z[0] ** 2 + z[2] ** 2
            x[0], x[2] = -z[2] / z_norm, z[0] / z_norm
            y = np.cross(z, x)
            y = y / np.linalg.norm(y)
            quat = quaternion.from_rotation_matrix(np.stack([x, y, z], axis=1))
            pc.vel_ctl(pos, quat, min_height=cfg('min_height'), ori_ctl=True)
            diff_height = np.abs(cfg('min_height') - pc.ee_pose()[0][2])
        pc.zeroize_vel()
        rospy.sleep(0.5)
        cpn_pub.publish(false)
        tsdf_save_pub.publish(true)
        tsdf_enable_pub.publish(false)
        pos, quat, width = pgpp.get()
        rot = quaternion.as_rotation_matrix(quat)
        if pos[2] < 0.17:
            z = np.array([0, 0, -1])
            y = rot[:, 1] - rot[:, 1] @ z * z
            y = y / np.linalg.norm(y)
            x = np.cross(y, z)
            rot = np.stack([x, y, z], axis=-1)
            quat = quaternion.from_rotation_matrix(rot)
        pre_pos = pos - rot[:, 2] * cfg('ee_offset')
        pos = pos + rot[:, 2] * 0.01
        ori_error = 1
        while ori_error > 0.005 and not pc.error:
            pc.vel_ctl(pre_pos, quat, ori_ctl=True)
            curr_pos, curr_quat = pc.ee_pose()
            quat_res = quaternion.as_float_array(quat * curr_quat.conj())
            ori_error = np.linalg.norm(quat_res[1:])
            print(ori_error)
        pc.zeroize_vel()
        rospy.sleep(0.5)
        # pc.movel(pre_pos, quat)
        angles = pc.angles()
        pc.set_gripper_width(width)
        rospy.sleep(0.5)
        diff_height = np.abs(pos[2] - pc.ee_pose()[0][2])
        while diff_height > 0.005 and not pc.error:
            pc.vel_ctl(pos, quat, min_height=pos[2], ori_ctl=False, gain_h=1)
            curr_pos = pc.ee_pose()[0]
            diff_height = np.abs(pos[2] - curr_pos[2])
        pc.zeroize_vel()
        rospy.sleep(0.5)
        pc.close_gripper()
        rospy.sleep(2.0)
        pc.movej(angles)
        if pc.detect_grasp():
            pc.movej(drop_pos)
        pc.open_gripper()
        pc.movej(cfg('jnt_vals'))
        rospy.sleep(0.5)


if __name__ == '__main__':
    panda_node()
