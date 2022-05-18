#!/usr/bin/env python3
from franka_core_msgs.msg import EndPointState
from vcpd.msg import EEPose
import rospy


class MessageConverter(object):
    def __init__(self):
        self.publisher = rospy.Publisher('/ee_pose', EEPose, queue_size=1)

    def callback(self, msg):
        ee = EEPose()
        ee.header = msg.header
        ee.pose = msg.O_T_EE
        self.publisher.publish(ee)


def main():
    topic_name = '/franka_ros_interface/custom_franka_state_controller/tip_state'
    rospy.init_node('pose_message_converter', anonymous=True)
    mc = MessageConverter()
    _ = rospy.Subscriber(topic_name, EndPointState, mc.callback, queue_size=1)
    rospy.loginfo('convert tip state to /ee_pose')
    rospy.spin()


if __name__ == '__main__':
    main()
