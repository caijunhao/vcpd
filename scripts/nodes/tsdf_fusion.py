#!/usr/bin/env python3
from sdf import SDF
from ros_utils import SDFCommander, Pose
from ros_utils import from_rs_intr_to_mat
from vcpd.srv import RequestVolume
from franka_core_msgs.msg import EndPointState
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
import pyrealsense2 as rs
import numpy as np
import torch
import message_filters
import threading
import rospy
import time
import copy
cfg = rospy.get_param


def tsdf_node():
    # initialize ros
    rospy.init_node('tsdf_node', anonymous=True)
    # initialize realsense
    if cfg('advanced'):
        dev = rs.context().query_devices()[0]
        advnc_mode = rs.rs400_advanced_mode(dev)
        depth_table_control_group = advnc_mode.get_depth_table()
        depth_table_control_group.disparityShift = cfg('disparity_shift')
        advnc_mode.set_depth_table(depth_table_control_group)
        rospy.loginfo('set disparity shift to {} in advanced mode'.format(cfg('disparity_shift')))
    pipeline = rs.pipeline()
    config = rs.config()
    rate = cfg('rate')
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, rate)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, rate)
    profile = pipeline.start(config)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    aligned = cfg('aligned')
    if aligned:
        rospy.loginfo('use aligned_to_color mode')
        intrinsic = from_rs_intr_to_mat(color_profile.get_intrinsics())
    else:
        rospy.loginfo('use direct mode')
        intrinsic = from_rs_intr_to_mat(depth_profile.get_intrinsics())
    align = rs.align(rs.stream.color)
    # warm up realsense
    for _ in range(rate):
        frames = pipeline.wait_for_frames()
        if aligned:
            aligned_frames = align.process(frames)
            _, _ = aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
        else:
            _, _ = frames.get_depth_frame(), frames.get_color_frame()

    # get transformation from end-effector to camera from rosparam
    if aligned:
        t_ee2cam = Pose(rospy.get_param('p_ee2cam_aligned'))
    else:
        t_ee2cam = Pose(rospy.get_param('p_ee2cam'))
    # tsdf initialization
    res = cfg('resolution')
    c_y = cfg('c_y_aligned') if aligned else cfg('c_y')
    vol_bnd = np.array([[cfg('c_x')-cfg('sdf_h')/2*res, c_y-cfg('sdf_w')/2*res, cfg('c_z')-cfg('sdf_d')/2*res],
                        [cfg('c_x')+cfg('sdf_h')/2*res, c_y+cfg('sdf_w')/2*res, cfg('c_z')+cfg('sdf_d')/2*res]])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        rospy.logwarn('no gpu found, use cpu for tsdf instead')
    tsdf = SDF(vol_bnd, res, rgb=True, device=device)
    pcl_pub = rospy.Publisher("/tsdf/pcl", PointCloud2, queue_size=1)
    sc = SDFCommander(tsdf, pcl_pub)
    # subscribe topics
    _ = rospy.Subscriber(cfg("ee_state"), EndPointState, sc.callback_t_base2ee, queue_size=1)
    _ = rospy.Subscriber("/tsdf/enable", Bool, sc.callback_enable)
    _ = rospy.Subscriber("/tsdf/reset", Bool, sc.callback_reset)
    _ = rospy.Subscriber("/tsdf/save", Bool, sc.callback_save)
    _ = rospy.Subscriber("/tsdf/pub_pcl", Bool, sc.callback_pcl)
    # add service to transmit sdf volume
    s = rospy.Service('request_a_volume', RequestVolume, sc.handle_send_tsdf_volume)
    # add threading lock to avoid refreshing the buffer while reading at the same time
    lock = threading.Lock()
    rospy.loginfo('tsdf_node is ready')
    noise_model = cfg('noise_model')
    a0, a1, a2 = cfg('a0'), cfg('a1'), cfg('a2')  # quadratic, linear, constant
    while not rospy.is_shutdown():
        if sc.start_flag:
            b = time.time()
            frames = pipeline.wait_for_frames()
            if aligned:
                aligned_frames = align.process(frames)
                depth_frame, color_frame = aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
            else:
                depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame:
                rospy.logwarn('no depth frame received')
                continue
            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data()).astype(np.float32) if cfg('use_color') else None
            lock.acquire()
            m_base2ee = copy.deepcopy(sc.t_base2ee())  # current pose from ee to base
            lock.release()
            m_base2cam = m_base2ee @ t_ee2cam()
            zero_flag = np.logical_or(depth < 100, depth > 1000)  # depth value is always smaller than 1000 in our task
            depth = depth / 1000.0
            if noise_model:
                depth = depth + np.exp(a0 * depth ** 2 + a1 * depth + a2)
            depth[zero_flag] = 0.0
            tsdf.integrate(depth, intrinsic, m_base2cam, rgb=color)
            rospy.loginfo("latest frame processed...")
            sc.valid_flag = True  # valid_flag will be true once visual data is feed into the volume
            e = time.time()
            elapse_time = e - b
            if 1 / elapse_time < rate:
                rospy.loginfo('process rate: {}Hz'.format(1 / elapse_time))
            else:
                rospy.sleep(1/rate-elapse_time)
                rospy.loginfo('process rate: {}Hz'.format(rate))
        if sc.reset_flag:
            tsdf.reset()
            sc.reset_flag = False
            sc.valid_flag = False
        if sc.save_flag:
            if sc.valid_flag:
                tsdf.write_mesh('out.ply', *tsdf.compute_mesh(gaussian_blur=True, step_size=3))
            else:
                rospy.logwarn('no visual data is merged into the volume currently, '
                              'no mesh will be saved at this moment.')
            sc.save_flag = False


if __name__ == '__main__':
    tsdf_node()