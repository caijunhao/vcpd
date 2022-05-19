#!/usr/bin/env python3
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
    pgpp = PandaGripperPoseParser('/cpn/gripper_pose')
    pc.movej(cfg('init_joint_pos'), unit='d')
    pc.open_gripper()
    drop_pos, drop_quat = pc.ee_pose()
    drop_pos[2] -= 0.3
    drop_pos[1] += 0.5
    tsdf_enable_pub = rospy.Publisher('/tsdf/enable', Bool, queue_size=1)
    tsdf_reset_pub = rospy.Publisher('/tsdf/reset', Bool, queue_size=1)
    tsdf_save_pub = rospy.Publisher('/tsdf/save', Bool, queue_size=1)
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
            pos, quat, _ = pgpp.get()
            pc.vel_ctl(pos, quat, cfg('min_height'))
            diff_height = np.abs(cfg('min_height') - pc.ee_pose()[0][2])
        pc.zeroize_vel()
        cpn_pub.publish(false)
        tsdf_enable_pub.publish(false)
        pos, quat, width = pgpp.get()
        rot = quaternion.as_rotation_matrix(quat)
        pre_pos = pos - rot[:, 2] * cfg('ee_offset')
        pc.movel(pre_pos, quat)
        pc.set_gripper_width(width)
        rospy.sleep(0.5)
        pc.movel(pos, quat)
        pc.movel(pre_pos, quat)
        pc.movel(drop_quat, quat)
        pc.open_gripper()
        pc.movej(cfg('init_joint_pos'), unit='d')


if __name__ == '__main__':
    panda_node()
