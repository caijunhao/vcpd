#!/usr/bin/env python3
from sdf import TSDF, PSDF, GradientSDF
from ros_utils import SDFCommander, Transform
from ros_utils import from_rs_intr_to_mat
from vcpd.srv import RequestVolume
from vcpd.msg import EEPose
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
import pyrealsense2 as rs
import numpy as np
import torch
import rospy
import time
cfg = rospy.get_param


def tsdf_node():
    # initialize ros
    rospy.init_node('tsdf_node', anonymous=True)
    # initialize realsense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, cfg('fps'))
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, cfg('fps'))
    profile = pipeline.start(config)
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    if cfg('aligned'):
        rospy.loginfo('use aligned_to_color mode')
        intrinsic = from_rs_intr_to_mat(color_profile.get_intrinsics())
    else:
        rospy.loginfo('use direct mode')
        intrinsic = from_rs_intr_to_mat(depth_profile.get_intrinsics())
    align = rs.align(rs.stream.color)
    # warm up realsense
    for _ in range(cfg('fps')):
        frames = pipeline.wait_for_frames()
        if cfg('aligned'):
            aligned_frames = align.process(frames)
            _, _ = aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
        else:
            _, _ = frames.get_depth_frame(), frames.get_color_frame()

    # get transformation from end-effector to camera from rosparam
    if cfg('aligned'):
        t_ee2cam = Transform(rospy.get_param('p_ee2cam_aligned'))
    else:
        t_ee2cam = Transform(rospy.get_param('p_ee2cam'))
    # sdf initialization
    vl = cfg('voxel_length')
    c_y = cfg('c_y_aligned') if cfg('aligned') else cfg('c_y')
    vol_bnd = np.array([[cfg('c_x')-cfg('sdf_h')/2*vl, c_y-cfg('sdf_w')/2*vl, cfg('c_z')-cfg('sdf_d')/2*vl],
                        [cfg('c_x')+cfg('sdf_h')/2*vl, c_y+cfg('sdf_w')/2*vl, cfg('c_z')+cfg('sdf_d')/2*vl]])
    resolution = np.ceil((vol_bnd[1] - vol_bnd[0] - vl / 7) / vl)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        rospy.logwarn('no gpu found, use cpu for sdf instead')
    sdf = TSDF(vol_bnd[0], resolution, vl, fuse_color=True, device=device)
    pcl_pub = rospy.Publisher("/sdf/pcl", PointCloud2, queue_size=1)
    # add threading lock to avoid refreshing the buffer while reading at the same time
    sc = SDFCommander(sdf, pcl_pub)
    # subscribe topics
    _ = rospy.Subscriber('/ee_pose', EEPose, sc.callback_t_base2ee, queue_size=1)
    _ = rospy.Subscriber("/sdf/enable", Bool, sc.callback_enable)
    _ = rospy.Subscriber("/sdf/reset", Bool, sc.callback_reset)
    _ = rospy.Subscriber("/sdf/save", Bool, sc.callback_save)
    _ = rospy.Subscriber("/sdf/pub_pcl", Bool, sc.callback_pcl)
    # add service to transmit sdf volume
    s = rospy.Service('request_a_volume', RequestVolume, sc.handle_send_tsdf_volume)
    rospy.loginfo('tsdf_node is ready')
    while not rospy.is_shutdown():
        if sc.start_flag:
            b = time.time()
            frames = pipeline.wait_for_frames()
            if cfg('aligned'):
                aligned_frames = align.process(frames)
                depth_frame, color_frame = aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
            else:
                depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame:
                rospy.logwarn('no depth frame received')
                continue
            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())[..., ::-1].copy() if cfg('use_color') else None
            m_base2cam = sc.m_base2ee @ t_ee2cam()
            zero_flag = np.logical_or(depth < 100, depth > 1000)  # depth value is always smaller than 1000 in our task
            depth = depth / 1000.0
            depth[zero_flag] = 0.0
            sc.tsdf_integrate(depth, intrinsic, m_base2cam, rgb=color)
            rospy.loginfo("latest frame processed...")
            sc.valid_flag = True  # valid_flag will be true once visual data is feed into the volume
            e = time.time()
            elapse_time = e - b
            if 1 / elapse_time < cfg('rate'):
                rospy.loginfo('process rate: {}Hz'.format(1 / elapse_time))
            else:
                rospy.sleep(1/cfg('rate')-elapse_time)
                rospy.loginfo('process rate: {}Hz'.format(cfg('rate')))
        if sc.reset_flag:
            sc.reset()
        if sc.save_flag:
            sc.save_mesh('/home/jcaiaq/out.ply')


if __name__ == '__main__':
    tsdf_node()
