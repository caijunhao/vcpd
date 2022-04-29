from utils import ImageProcessor
from torch.nn.functional import conv3d, grid_sample
from skimage import measure
import open3d as o3d
import numpy as np
import torch
import time


class SDF(object):
    def __init__(self, volume_bounds, resolution, rgb=False, device='cuda', dtype=torch.float32):
        """
        Initialize SDF module
        :param volume_bounds: A 2x3-D numpy array specifying the min and max bound values of the volume for each axis
        :param resolution: The res for each voxel, the unit is meter.
        :param rgb: A flag specifying if fusing color information
        :param device: An object representing the dev on which a torch.Tensor is or will be allocated.
        :param dtype: Data type for torch tensor.
        """
        self.vol_bnd = volume_bounds.astype(np.float32)
        self.res = resolution
        self.margin = resolution * 5
        self.max_w = 64
        self.rgb = rgb
        self.dev = device
        self.dtype = dtype

        # Adjust vol_bnd
        # minus a small value to avoid float data error
        self.vol_bnd[1] = self.vol_bnd[1] - self.res / 7
        self.vol_dims = np.ceil((self.vol_bnd[1] - self.vol_bnd[0]) / self.res).astype(int)
        self.vol_bnd[1] = self.vol_bnd[0] + self.vol_dims * self.res
        self.origin = torch.from_numpy(self.vol_bnd[0]).view(1, 3).to(dtype).to(device)

        # Initialize signed distance and weight volumes
        self.sdf_vol = torch.ones(self.vol_dims.tolist(), dtype=self.dtype, device=device)
        # assign a small value to avoid zero division
        self.w_vol = torch.ones_like(self.sdf_vol, dtype=self.dtype, device=device) * 1e-7
        # record if the voxel has been scan at least once
        self.pos_vol = torch.zeros_like(self.sdf_vol, dtype=torch.bool, device=device)  # positive-value voxels
        if self.rgb:
            self.rgb_vol = torch.ones_like(self.sdf_vol, dtype=self.dtype, device=device)
            self.rgb_vol = self.rgb_vol * (255 * 256 ** 2 + 255 * 256 + 255)

        # get voxel coordinates and world positions
        vx, vy, vz = torch.meshgrid(torch.arange(self.vol_dims[0]),
                                    torch.arange(self.vol_dims[1]),
                                    torch.arange(self.vol_dims[2]))
        self.voxel_ids = torch.stack([vx, vy, vz], dim=-1).to(device)
        voxel_pos = self.origin + self.voxel_ids * self.res
        self.voxel_pos = torch.cat([voxel_pos,
                                    torch.ones(self.vol_dims.tolist() + [1], dtype=dtype).to(device)],
                                   dim=-1)  # homogenous representation of voxel positions in world frame
        self.sdf_info()
        self.ip = ImageProcessor()

    def integrate(self, depth, intrinsic, camera_pose, fd='point2point', fw='etw', rgb=None):
        """
        Integrate RGB-D frame into SDF volume (naive SDF)
        :param depth: An HxW numpy array representing a depth map
        :param intrinsic: A 3x3 numpy array representing the camera intrinsic matrix
        :param camera_pose: A 4x4 numpy array representing the transformation from reference frame (world) to camera
        :param fd: name of the distance function used for sdf integration.
        :param fw: name of the weighting function used for sdf integration.
        :param rgb: (Optional) An HxWx3 numpy array representing a color image
        :return: None
        """
        depth = self.tensorize(depth)
        w2c = self.tensorize(camera_pose)
        intrinsic = self.tensorize(intrinsic)
        height, width = depth.shape
        c2w = torch.inverse(w2c)
        voxel_pos_c = self.voxel_pos @ c2w.T  # position represented in camera frame
        voxel_ids = voxel_pos_c[..., :-1] @ intrinsic.T
        # we assume that the camera is always outside the volume space and directed toward the volume,
        # so all the z value should larger than 0
        flag_z = voxel_ids[..., 2] > 0
        voxel_ids[flag_z, :] = voxel_ids[flag_z, :] / voxel_ids[flag_z, :][:, -1:]
        flag_w = (voxel_ids[..., 0] >= 0) * (voxel_ids[..., 0] < width - 0.5)
        flag_h = (voxel_ids[..., 1] >= 0) * (voxel_ids[..., 1] < height - 0.5)
        voxel_flag = flag_w * flag_h * flag_z  # dim0 x dim1 x dim2
        valid_ids = torch.stack([voxel_ids[voxel_flag][:, 1], voxel_ids[voxel_flag][:, 0]], dim=-1)
        valid_pos = voxel_pos_c[voxel_flag]
        distance, depth_flag = self.__getattribute__(fd)(depth, valid_ids, valid_pos, intrinsic)
        pos_flag = distance >= 0
        pos_x, pos_y, pos_z = self.voxel_ids[voxel_flag][depth_flag][pos_flag].unbind(dim=-1)
        self.pos_vol[pos_x, pos_y, pos_z] = True
        # self.neg_vol[voxel_flag][depth_flag] = distance < 0
        valid_x, valid_y, valid_z = self.voxel_ids[voxel_flag][depth_flag].unbind(dim=-1)
        # update sdf volume with given weighting function
        w_old = self.w_vol[valid_x, valid_y, valid_z]
        sdf_old = self.sdf_vol[valid_x, valid_y, valid_z]
        w_curr = self.__getattribute__(fw)(distance)
        w_new = w_old + w_curr
        sdf_new = (w_old * sdf_old + w_curr * distance) / w_new
        w_new = torch.clamp_max(w_new, self.max_w)
        self.w_vol[valid_x, valid_y, valid_z] = w_new
        self.sdf_vol[valid_x, valid_y, valid_z] = sdf_new
        if self.rgb and rgb is not None:
            rgb_old = self.decode_rgb(self.rgb_vol[valid_x, valid_y, valid_z])
            rgb = torch.tensor(rgb, self.dtype, self.dev) if isinstance(rgb, np.ndarray) else rgb.clone().to(
                self.dtype).to(self.dev)
            valid_ids = torch.round(valid_ids).long()
            valid_color = rgb[valid_ids[:, 0], valid_ids[:, 1]]
            rgb_new = (w_old.unsqueeze(dim=1) * rgb_old + w_curr.unsqueeze(dim=1) * valid_color) / w_new.unsqueeze(
                dim=1)
            rgb_new = torch.clamp_max(rgb_new, 255)
            self.rgb_vol[valid_x, valid_y, valid_z] = self.encode_rgb(rgb_new)

    def point2point(self, depth, depth_ids, voxel_pos, *args, **kwargs):
        # depths = self.ip.interpolate(depth, depth_ids)
        depth_ids = torch.round(depth_ids).long()
        depths = depth[depth_ids[:, 0], depth_ids[:, 1]]
        valid_depth = depths > 0
        distance = depths[valid_depth] - voxel_pos[valid_depth][:, 2]
        distance = distance / self.margin
        return distance, valid_depth

    def point2plane(self, depth, depth_ids, voxel_pos, intrinsic):
        depths = self.ip.interpolate(depth, depth_ids)
        valid_depth = depths > 0
        pixel_pos = self.ip.pixel2point(depth, intrinsic)
        pixel_pos = self.ip.interpolate(pixel_pos, depth_ids)
        x, y = voxel_pos[valid_depth][..., 0:3], pixel_pos[valid_depth]
        normal = self.ip.normal_estimation(depth, intrinsic)
        # directly interpolating normal is infeasible,
        # we instead assign the normal of the nearest neighbor to each query index.
        round_ids = torch.round(depth_ids).to(torch.long)
        n = normal[round_ids[:, 0], round_ids[:, 1]]
        distance = torch.sum((x - y) * n, dim=1)
        distance = distance / self.margin
        return distance, valid_depth

    def reset(self):
        self.sdf_vol = torch.ones_like(self.sdf_vol, dtype=self.dtype, device=self.dev)
        self.w_vol = torch.ones_like(self.sdf_vol, dtype=self.dtype, device=self.dev) * 1e-7
        self.pos_vol = torch.zeros_like(self.sdf_vol, dtype=torch.bool, device=self.dev)
        if self.rgb:
            self.rgb_vol = torch.ones_like(self.sdf_vol, dtype=self.dtype, device=self.dev)
            self.rgb_vol = self.rgb_vol * (255 * 256 ** 2 + 255 * 256 + 255)

    def tensorize(self, data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=self.dtype, device=self.dev)
        else:
            data = data.clone().to(self.dtype).to(self.dev)
        return data

    @property
    def shape(self):
        return self.sdf_vol.shape

    @staticmethod
    def cw(distance):
        """
        constant weight
        :param distance:
        :return:
        """
        w = torch.zeros_like(distance)
        flag = distance <= 1
        w[flag] = 1
        return w

    @staticmethod
    def ncw(distance):
        """
        narrow constant weight
        :param distance:
        :return:
        """
        w = torch.zeros_like(distance)
        flag = torch.logical_and(distance >= -1, distance <= 1)
        w[flag] = 1
        return w

    @staticmethod
    def nlw(distance, eps=0.3):
        """
        narrow linear weight
        :param distance:
        :param eps: penetration depth.
        :return:
        """
        assert 0.0 <= eps <= 1.0
        w = torch.zeros_like(distance)
        flag_const = torch.logical_and(distance >= -eps, distance <= eps)
        flag_linear = torch.logical_or(torch.logical_and(distance >= -1, distance <= -eps),
                                       torch.logical_and(distance >= eps, distance <= 1))
        w[flag_const] = 1
        w[flag_linear] = (1 - distance[flag_linear]) / (1 - eps)
        return w

    @staticmethod
    def new(distance, eps=0.3, sigma=7):
        """
        narrow exponential weight
        :param distance:
        :param eps: penetration depth.
        :param sigma:
        :return:
        """
        assert 0.0 <= eps <= 1.0
        w = torch.zeros_like(distance)
        flag_exp = torch.logical_or(torch.logical_and(distance >= -1, distance <= -eps),
                                    torch.logical_and(distance >= eps, distance <= 1))
        flag_const = torch.logical_and(distance >= -eps, distance <= eps)
        w[flag_const] = 1
        w[flag_exp] = torch.exp(- sigma * (torch.abs(distance[flag_exp]) - eps) ** 2)
        return w

    @staticmethod
    def etw(distance, eps=0.3):
        """
        early truncated weight
        :param distance:
        :param eps: penetration depth.
        :return:
        """
        assert 0.0 <= eps <= 1.0
        w = torch.zeros_like(distance)
        flag_const = torch.logical_and(distance >= -eps, distance <= eps)
        w[flag_const] = 1
        return w

    @property
    def post_processed_volume(self):
        """
        Process SDF volume before computing mesh or point cloud.
        Should be overwritten in subclass.
        :return: None
        """
        # if the initial view frustum includes the entire volume space,
        # we can use weight volume to find occupied space
        sdf_vol = self.sdf_vol.clone()
        in_obj_ids = torch.logical_not(self.pos_vol) * (self.sdf_vol == 1)
        sdf_vol[in_obj_ids] = -1.0
        return sdf_vol

    def compute_pcl(self, threshold=0.2, use_post_processed=True, gaussian_blur=True):
        b = time.time()
        if use_post_processed:
            sdf_volume = self.post_processed_volume
        else:
            sdf_volume = self.sdf_vol
        if gaussian_blur:
            sdf_volume = self.gaussian_blur(sdf_volume, device=self.dev)
        valid_voxels = torch.abs(sdf_volume) < threshold
        # valid_voxels = curr_sdf_volume < threshold
        xyz = self.voxel_pos[valid_voxels]
        if self.rgb:
            rgb = self.decode_rgb(self.rgb_vol[valid_voxels])
        else:
            rgb = (torch.ones_like(xyz, dtype=self.dtype) * 255).to(self.dev)
        pcl = torch.cat([xyz, rgb], dim=1)
        e = time.time()
        print('elapse time for computing point cloud: {:06f}s'.format(e - b))
        return pcl

    def compute_mesh(self, use_post_processed=True, gaussian_blur=True, step_size=3):
        """
        Using marching cubes to compute a mesh from SDF volume.
        https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html
        :param use_post_processed: A boolean flag to specify if use post-processed volume
        :param gaussian_blur: A boolean flag specifying whether to do gaussian smoothing for the volume
        :param step_size: Step size in voxels.
        :return: Vertices, faces, normals, and rgb values of the constructed mesh.
        """
        b = time.time()
        if use_post_processed:
            sdf_volume = self.post_processed_volume
        else:
            sdf_volume = self.sdf_vol
        if gaussian_blur:
            sdf_volume = self.gaussian_blur(sdf_volume, device=self.dev)

        sdf_volume = sdf_volume.cpu().numpy()
        v, f, n, _ = measure.marching_cubes(sdf_volume, level=0, step_size=step_size)
        n = -n
        ids = np.round(v).astype(np.int)
        v = v * self.res + self.vol_bnd[0].reshape(1, 3)
        if self.rgb:
            rgb = self.decode_rgb(self.rgb_vol[ids[:, 0], ids[:, 1], ids[:, 2]])
            rgb = rgb.cpu().numpy()
            rgb = np.floor(rgb)
            rgb = rgb.astype(np.uint8)
        else:
            rgb = np.ones_like(v, dtype=np.uint8) * 255
        e = time.time()
        print('elapse time for marching cubes: {:06f}s'.format(e - b))
        return v, f, n, rgb

    def extract_sdf(self, pos, post_processed=True, gaussian_blur=True, mode='bilinear', padding_mode='border'):
        """
        Extract sdf values given query positions in the volume (world) frame.
        :param pos: An Nx3-D numpy array or torch tensor representing the 3D position in the volume (world) frame.
        :param post_processed: Whether to process the sdf volume or not before extracting the sdf values.
        :param gaussian_blur:
        :param mode: mode of interpolation.
        :param padding_mode: the way of padding mode
        :return: An N-D torch tensor representing the retrieved sdf values.
        """
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos).to(self.dtype).to(self.dev)
        num_pos = pos.shape[0]
        pos = torch.transpose(pos, 1, 0).view(1, 3, num_pos, 1)
        res = torch.tensor(self.res).to(self.dev).view(1, 1, 1, 1)
        ids = (pos - self.origin.view(1, 3, 1, 1)) / res
        sdf_vol = self.post_processed_volume if post_processed else self.sdf_vol
        if gaussian_blur:
            sdf_vol = self.gaussian_blur(sdf_vol)
        h, w, d = sdf_vol.shape
        sdf_vol = sdf_vol.view(1, 1, h, w, d)
        size = torch.from_numpy(np.array([d - 1, w - 1, h - 1], dtype=np.float32).reshape((1, 1, 1, 1, 3))).to(
            self.dev)
        indices = torch.unsqueeze(ids.permute(0, 2, 3, 1), dim=3)  # B*N*1*1*3
        indices = torch.stack([indices[..., 2], indices[..., 1], indices[..., 0]], dim=-1)
        grid = indices / size * 2 - 1  # [-1, 1]
        output = grid_sample(sdf_vol, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
        output = torch.squeeze(output)
        return output

    def get_ids(self, pos):
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos).to(self.dev)
        res = torch.tensor(self.res).to(self.dev)
        ids = (pos - self.origin) / res
        return ids

    def sdf_info(self):
        print('volume bounds: ')
        print('x_min: {:04f}m, y_min: {:04f}m, z_min: {:04f}m'.format(*self.vol_bnd[0].tolist()))
        print('x_max: {:04f}m, y_max: {:04f}m, z_max: {:04f}m'.format(*self.vol_bnd[1].tolist()))
        print('volume size: {}'.format(self.vol_dims))
        print('res: {:06f}m'.format(self.res))

    @property
    def gradients(self):
        """
        Return the gradient of SDF volume.
        The gradient is computed using second order accurate central differences in the interior points and
        first order accurate one-sides differences at the boundaries.
        For more details, you can refer to this link.
        https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
        :param sdf_volume: An SDF volume
        :param device: The desired dev of returned tensor.
        :return: A torch tensor with the same shape as self.world_pos.
                 Each channel represents one derivative of the SDF volume
        """
        sdf_vol = self.post_processed_volume
        dtype = sdf_vol.dtype
        dev = sdf_vol.device
        h, w, d = sdf_vol.size()
        gradients = torch.zeros([h, w, d, 3], dtype=dtype, device=dev)
        # interior points
        gradients[1:h - 1, ..., 0] = sdf_vol[2:h, ...] - sdf_vol[0:h - 2, ...]
        gradients[:, 1:w - 1, :, 1] = sdf_vol[:, 2:w, ...] - sdf_vol[:, 0:w - 2, ...]
        gradients[..., 1:d - 1, 2] = sdf_vol[..., 2:d] - sdf_vol[..., 0:d - 2]
        gradients = gradients / 2.0
        # boundaries
        gradients[0, ..., 0] = sdf_vol[1, ...] - sdf_vol[0, ...]
        gradients[-1, ..., 0] = sdf_vol[-1, ...] - sdf_vol[-2, ...]
        gradients[:, 0, :, 1] = sdf_vol[:, 1, ...] - sdf_vol[:, 0, ...]
        gradients[:, -1, :, 1] = sdf_vol[:, -1, ...] - sdf_vol[:, -2, ...]
        gradients[..., 0, 2] = sdf_vol[..., 1] - sdf_vol[..., 0]
        gradients[..., -1, 2] = sdf_vol[..., -1] - sdf_vol[..., -2]
        return gradients

    @staticmethod
    def write_mesh(filename, vertices, faces, normals, rgbs):
        """
        Save a 3D mesh to a polygon .ply file.
        """
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (vertices.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("element face %d\n" % (faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")
        # Write vertex list
        for i in range(vertices.shape[0]):
            ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
                vertices[i, 0], vertices[i, 1], vertices[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                rgbs[i, 0], rgbs[i, 1], rgbs[i, 2],))
        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))
        ply_file.close()

    @staticmethod
    def write_pcl(filename, pcl):
        """
        Save a point cloud to a polygon .ply file.
        :param filename: The file name of the.ply file.
        :param pcl: An N x 6 torch tensor representing the point cloud with rgb information.
        :return: None
        """
        pcl = pcl.cpu().numpy()
        xyz = pcl[:, :3]
        rgb = pcl[:, 3:].astype(np.uint8)
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (xyz.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        # Write vertex list
        for i in range(xyz.shape[0]):
            ply_file.write("%f %f %f %d %d %d\n" % (
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                rgb[i, 0], rgb[i, 1], rgb[i, 2],))

    @staticmethod
    def decode_rgb(rgb_volume):
        """
        Decode an RGB volume to R, G, and B volumes separately
        :param rgb_volume: An N-d torch tensor in which each voxel encodes the RGB information as B * 256 * 256 + G * 256 + R
        :return: An N x 3 torch tensors representing the R, G, B values of selected pixels of the color image
        """
        r = rgb_volume % 256
        g = (rgb_volume - r) / 256 % 256
        b = ((rgb_volume - r) / 256 - g) / 256
        return torch.stack([r, g, b], dim=1)

    @staticmethod
    def encode_rgb(rgb):
        """
        Encode each pixel of color image as B * 256 * 256 + G * 256 + R
        :param rgb: An N x 3 torch tensor representing the R, G, B values of selected pixels of the color image
        :return: An N-d torch tensor
        """
        return rgb[:, 2] * 256 * 256 + rgb[:, 1] * 256 + rgb[:, 0]

    @staticmethod
    def gaussian_blur(volume, kernel_size=3, sigma_square=0.7, device='cuda'):
        if len(volume.shape) != 5:
            volume = torch.unsqueeze(torch.unsqueeze(volume, dim=0), dim=0)
        h = kernel_size // 2
        kernel_ids = torch.arange(kernel_size, device=device, dtype=torch.float32)
        x, y, z = torch.meshgrid(kernel_ids, kernel_ids, kernel_ids)
        kernel = torch.exp(-((x - h) ** 2 + (y - h) ** 2 + (z - h) ** 2) / (2 * sigma_square))
        kernel /= torch.sum(kernel)
        kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=0)
        volume = conv3d(volume, weight=kernel, stride=1, padding=h)
        volume = torch.squeeze(volume)
        return volume


class TSDF(object):
    def __init__(self, volume_bounds, voxel_length, rgb=False):
        """
        Initialize SDF module
        :param volume_bounds: A 2x3-D numpy array specifying the min and max bound values of the volume for each axis
        :param voxel_length: The voxel length for each voxel, the unit is meter.
        :param rgb: A flag specifying if fusing color information
        :param dtype: Data type for torch tensor.
        """
        self.origin = volume_bounds[0]
        volume_bounds[1] = volume_bounds[1] - voxel_length / 7
        self.vol_dims = np.ceil((volume_bounds[1] - volume_bounds[0]) / voxel_length).astype(int)
        volume_bounds[1] = volume_bounds[0] + self.vol_dims * voxel_length
        self.vol_bnds = volume_bounds
        resolution = np.max(self.vol_dims)
        length = voxel_length * resolution
        self.rgb = rgb
        if rgb:
            color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        else:
            color_type = o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        self.vol = o3d.pipelines.integration.UniformTSDFVolume(length=length,
                                                               resolution=resolution,
                                                               sdf_trunc=voxel_length*4,
                                                               color_type=color_type,
                                                               origin=self.origin.reshape(-1, 1))
        self.num_integrate = 0

    def integrate(self, depth, intrinsic, camera_pose, rgb=None):
        """
        Integrate RGB-D frame into SDF volume
        :param depth: An HxW numpy array representing a depth map
        :param intrinsic: A 3x3 numpy array representing the camera intrinsic matrix
        :param camera_pose: A 4x4 numpy array representing the transformation from camera to reference frame (world)
        :param rgb: (Optional) An HxWx3 numpy array representing a color image
        :return: None
        """
        height, width = depth.shape
        depth = o3d.geometry.Image(depth.astype(np.float32))
        rgb = o3d.geometry.Image(rgb) if self.rgb and rgb is not None else o3d.geometry.Image(np.empty_like(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,
                                                                  depth_scale=1.0,
                                                                  depth_trunc=2.0,
                                                                  convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(width=width, height=height,
                                                 fx=intrinsic[0, 0], fy=intrinsic[1, 1],
                                                 cx=intrinsic[0, 2], cy=intrinsic[1, 2])
        self.vol.integrate(rgbd, intr, np.linalg.inv(camera_pose))
        self.num_integrate += 1

    def reset(self):
        self.vol.reset()
        self.num_integrate = 0

    def extract_mesh(self):
        return self.vol.extract_triangle_mesh()

    def write_triangle_mesh(self, filename):
        o3d.io.write_triangle_mesh(filename, self.vol.extract_triangle_mesh())

    def extract_vol(self):
        h, w, d = self.vol_dims
        res = self.vol.resolution
        data = np.asarray(self.vol.extract_volume_tsdf()).reshape((res, res, res, 2))
        tsdf, w_vol = data[..., 0], data[..., 1]
        return tsdf[0:h, 0:w, 0:d], w_vol[0:h, 0:w, 0:d]

    def extract_mesh2(self, step_size=3):
        tsdf, w = self.extract_vol()
        voxel_length = self.vol.voxel_length
        v, f, n, _ = measure.marching_cubes(tsdf, level=0, step_size=step_size)
        n = -n
        ids = np.round(v).astype(int)
        v = v * voxel_length + self.origin.reshape(1, -1)
        if self.rgb:
            res = self.vol.resolution
            rgb = np.asarray(self.vol.extract_volume_color().reshape((res, res, res, 3)))
            rgb = rgb[ids[:, 0], ids[:, 1], ids[:, 2]]
        else:
            rgb = np.ones_like(v, dtype=np.uint8) * 255
        return v, f, n, rgb

    def extract_sdf(self, coordinate):
        if coordinate.ndim == 1:
            coordinate = coordinate.reshape(1, 3)
            squeeze = True
        else:
            squeeze = False
        tsdf = self.tsdf
        ids = np.round((coordinate - self.origin.reshape(-1, 3)) / self.vol.voxel_length)
        ids = ids.astype(int)
        flag = ids >= self.vol_dims.reshape(1, 3)
        for i in range(3):
            ids[:, i][flag[:, i]] = self.vol_dims[i] - 1
        out = np.squeeze(tsdf[ids[:, 0], ids[:, 1], ids[:, 2]]) if squeeze else tsdf[ids[:, 0], ids[:, 1], ids[:, 2]]
        return out

    def get_ids(self, coordinate):
        ids = (coordinate - self.origin.reshape(-1, 3)) / self.vol.voxel_length
        return ids

    @property
    def tsdf(self):
        tsdf, _ = self.extract_vol()
        return tsdf

    @property
    def w(self):
        _, w = self.extract_vol()
        return w

    @property
    def rgb_vol(self):
        if self.rgb:
            res = self.vol.resolution
            rgb = np.asarray(self.vol.extract_volume_color().reshape((res, res, res, 3)))
            return rgb
        else:
            return None

    @property
    def gradient(self):
        res = self.vol.resolution
        tsdf = self.tsdf
        gradient = np.zeros([res, res, res, 3])
        # central difference fro interior points
        gradient[1:res-1, ..., 0] = tsdf[2:res, ...] - tsdf[0:res-2, ...]
        gradient[:, 1:res-1, :, 1] = tsdf[:, 2:res, ...] - tsdf[:, 0:res-2, ...]
        gradient[..., 1:res-1, 2] = tsdf[..., 2:res] - tsdf[..., 0:res-2]
        gradient = gradient / 2.0
        # boundaries
        gradient[0, ..., 0] = tsdf[1, ...] - tsdf[0, ...]
        gradient[-1, ..., 0] = tsdf[-1, ...] - tsdf[-2, ...]
        gradient[:, 0, :, 1] = tsdf[:, 1, ...] - tsdf[:, 0, ...]
        gradient[:, -1, :, 1] = tsdf[:, -1, ...] - tsdf[:, -2, ...]
        gradient[..., 0, 2] = tsdf[..., 1] - tsdf[..., 0]
        gradient[..., -1, 2] = tsdf[..., -1] - tsdf[..., -2]
        return gradient

    @property
    def voxel_coordinate(self):
        res = self.vol.resolution
        voxel_length = self.vol.voxel_length
        vx, vy, vz = np.meshgrid(np.arange(res), np.arange(res), np.arange(res), indexing='ij')
        voxel_ids = np.stack([vx, vy, vz], axis=-1)
        voxel_coordinate = self.origin.reshape(1, 1, 1, 3) + voxel_ids * voxel_length
        return voxel_coordinate

    @staticmethod
    def write_mesh(filename, vertices, faces, normals, rgbs):
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (vertices.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("element face %d\n" % (faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")
        # Write vertex list
        for i in range(vertices.shape[0]):
            ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
                vertices[i, 0], vertices[i, 1], vertices[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                rgbs[i, 0], rgbs[i, 1], rgbs[i, 2],))
        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))
        ply_file.close()
