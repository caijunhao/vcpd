#!/usr/bin/env python3
from std_msgs.msg import Bool
from vcpd.msg import GripperPose
from ros_utils import CPNCommander
from cpn.model import CPN
import numpy as np
import pymeshlab as ml
import rospy
import torch
import time
import os
cfg = rospy.get_param


def cpn_node():
    rospy.init_node('cpn_node', anonymous=True)
    # Initialize CPNCommander
    aligned = cfg('aligned')
    c_y = cfg('c_y_aligned') if aligned else cfg('c_y')
    vl = cfg('voxel_length')
    vol_bnd = np.array([[cfg('c_x')-cfg('sdf_h')/2*vl, c_y-cfg('sdf_w')/2*vl, cfg('c_z')-cfg('sdf_d')/2*vl],
                        [cfg('c_x')+cfg('sdf_h')/2*vl, c_y+cfg('sdf_w')/2*vl, cfg('c_z')+cfg('sdf_d')/2*vl]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPN()
    path = cfg('model_path')
    if path:
        model.load_network_state_dict(device=device, pth_file=path)
        model.to(device)
        rospy.loginfo('successfully load model from: {}'.format(path))
    else:
        rospy.logwarn('no pretrained model ...')
    ms = ml.MeshSet()
    vertex_sets = dict()
    for component in ['hand', 'left_finger', 'right_finger']:
        col2_path = os.path.join(cfg('asset_path'), component + '_col2.obj')
        ms.load_new_mesh(col2_path)
        vertex_sets[component] = ms.current_mesh().vertex_matrix()
    vc = CPNCommander(vol_bnd, vl, model, vertex_sets, gripper_depth=cfg('gripper_depth'), device=device)
    _ = rospy.Subscriber("/cpn/flag", Bool, vc.callback_cpn)
    gripper_pose_pub = rospy.Publisher('/cpn/gripper_pose', GripperPose, queue_size=1)
    rospy.loginfo('cpn_node is ready')
    while not rospy.is_shutdown():
        if vc.trigger_flag:
            b = time.time()
            msg = vc.inference()
            if msg is not None:
                gripper_pose_pub.publish(msg)
            e = time.time()
            elapse_time = e - b
            rospy.sleep(max(0.0, 1/30-elapse_time))


if __name__ == '__main__':
    cpn_node()
