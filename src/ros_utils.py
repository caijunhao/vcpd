from scipy.spatial.transform.rotation import Rotation
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import rospy


def from_rs_intr_to_mat(rs_intrinsics):
    intrinsic = np.array([rs_intrinsics.fx, 0.0, rs_intrinsics.ppx,
                          0.0, rs_intrinsics.fy, rs_intrinsics.ppy,
                          0.0, 0.0, 1.0]).reshape(3, 3)
    return intrinsic


class Pose(object):
    def __init__(self, array):
        array = np.asanyarray(array)
        dtype = array.dtype
        if array.shape[0] == 7:
            self.pos_n_quat = array
            self.pose = np.eye(4, dtype=dtype)
            self.pose[0:3, 0:3] = Rotation.from_quat(array[3:7]).as_matrix()
            self.pose[0:3, 3] = array[0:3]
        elif array.shape[0] == 4 and array.shape[1] == 4:
            self.pose = array
            self.pos_n_quat = np.concatenate([array[0:3], Rotation.from_matrix(array[0:3, 0:3]).as_quat()])

    def __call__(self):
        return self.pose


class SDFCommander(object):
    def __init__(self, tsdf, pcl_pub):
        self.tsdf = tsdf
        self.pcl_pub = pcl_pub

        self.t_base2ee = None  # current pose from base frame to O_T_EE
        self.processed_flag = True  # flag to specify whether current visual data in this buffer is processed or not
        self.num_received = 0  # number of visual data received

        self.start_flag = False  # trigger flag to specify whether to receive data or not
        self.save_flag = False  # flag to save the mesh generated from sdf volume
        self.reset_flag = False  # flag to reset values for the sdf volume
        self.valid_flag = False  # flag to specify if it is valid to response to the request

    def callback_t_base2ee(self, msg):
        ee_pose = np.asarray(msg.O_T_EE).reshape((4, 4), order='F').astype(np.float32)
        self.t_base2ee = Pose(ee_pose)

    def callback_enable(self, msg):
        """
        :param msg: vpn/Bool.msg
        """
        self.start_flag = msg.data
        if self.start_flag:
            rospy.loginfo("flag received, now start processing tsdf")
        else:
            rospy.loginfo("flag received, pause processing")

    def callback_save(self, msg):
        """
        :param msg: vpn/Bool.msg
        """
        self.save_flag = msg.data
        rospy.loginfo("flag received, generate mesh from tsdf and save to the disk")

    def callback_reset(self, msg):
        """
        :param msg: vpn/Bool.msg
        """
        self.reset_flag = msg.data
        rospy.loginfo("flag received, re-initialize tsdf")

    def handle_send_tsdf_volume(self, req):
        while not self.valid_flag:
            rospy.sleep(0.01)
        if req.post_processed:
            sdf_volume = self.tsdf.post_processed_volume
        else:
            sdf_volume = self.tsdf.sdf_volume
        if req.gaussian_blur:
            sdf_volume = self.tsdf.gaussian_blur(sdf_volume)
        sdf_volume = sdf_volume.cpu().numpy()
        volume_msg = Volume()
        volume_msg.height, volume_msg.width, volume_msg.depth = sdf_volume.shape
        volume_msg.data = sdf_volume.reshape(-1).tolist()
        return RequestVolumeResponse(volume_msg)

    def callback_pcl(self, msg):
        while not self.valid_flag:
            rospy.loginfo('wait for visual data ...')
            rospy.sleep(0.1)
            return
        if msg.data:
            xyz_rgb = self.tsdf.compute_pcl(use_post_processed=False, gaussian_blur=False)
            xyz_rgb = xyz_rgb.cpu().numpy()
            pcl2msg = self.array_to_pointcloud2(xyz_rgb)
            self.pcl_pub.publish(pcl2msg)
        rospy.sleep(0.1)

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
