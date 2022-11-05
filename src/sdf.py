from skimage import measure
from torch.nn.functional import conv2d, conv3d, grid_sample
import numpy as np
import torch
import time


class TSDF(object):
    def __init__(self, origin, resolution, voxel_length, truncated_length=0.005,
                 fuse_color=False, device='cuda', dtype=torch.float):
        """
        Initialize SDF module
        :param origin: (ndarray) [3, ] the world coordinate of the voxel [0, 0, 0]
        :param resolution: (ndarray) [3, ] resolution over the total length of the volume
        :param voxel_length: (float) voxel size, where length = voxel_length * resolution
        :param truncated_length: (float) the length of the margin
        :param fuse_color: (bool) A flag specifying if fusing color information or not
        :param device: An object representing the device on which a torch.Tensor is or will be allocated.
        :param dtype: Data type for torch tensor.
        """
        self.origin = torch.from_numpy(origin).to(device).to(dtype)
        self.res = torch.from_numpy(resolution).to(torch.long).to(device)
        self.vox_len = torch.tensor(voxel_length, dtype=dtype, device=device)
        self.sdf_trunc = truncated_length
        self.fuse_color = fuse_color
        self.dev = device
        self.dt = dtype

        # Initialize volumes
        self.sdf_vol = torch.ones(*self.res).to(self.dt).to(self.dev)
        self.w_vol = torch.zeros(*self.res).to(self.dt).to(self.dev)
        self.grad_vol = None
        self.aux_vol = torch.zeros(*self.res).to(torch.bool).to(self.dev)
        if self.fuse_color:
            self.rgb_vol = torch.zeros(*self.res, 3).to(self.dt).to(self.dev)

        # get voxel coordinates and world coordinates
        vx, vy, vz = torch.meshgrid(torch.arange(self.res[0]), torch.arange(self.res[1]), torch.arange(self.res[2]),
                                    indexing='ij')
        self.vox_coords = torch.stack([vx, vy, vz], dim=-1).to(device)
        self.world_coords = self.vox_coords * self.vox_len + self.origin.view(1, 1, 1, 3)
        self.world_coords = torch.cat([self.world_coords,
                                       torch.ones(*self.res, 1).to(self.dt).to(self.dev)],
                                      dim=-1)
        self.sdf_info()

    def integrate(self, depth, intrinsic, camera_pose, dist_func='point2point', rgb=None):
        """
        Integrate RGB-D frame into SDF volume (naive SDF)
        :param depth: (ndarray.float) [H, W] a depth map whose unit is meter.
        :param intrinsic: (ndarray.float) [3, 3] the camera intrinsic matrix
        :param camera_pose: (ndarray.float) [4, 4] the transformation from world to camera frame
        :param dist_func: (str) point2point or point2plane
        :param rgb: (ndarray.uint8) [H, W, 3] a color image
        :return: None
        """
        depth = self.to_tensor(depth)
        cam_intr = self.to_tensor(intrinsic)
        fx, fy, cx, cy = cam_intr[0, 0], cam_intr[1, 1], cam_intr[0, 2], cam_intr[1, 2]
        cam_pose = self.to_tensor(camera_pose)  # w2c
        im_h, im_w = depth.shape
        c2w = torch.inverse(cam_pose)
        cam_coords = self.world_coords @ c2w.T  # world coordinates represented in camera frame
        pix_z = cam_coords[..., 2]
        # project all the voxels back to image plane
        pix_x = torch.round((cam_coords[..., 0] * fx / cam_coords[..., 2]) + cx).long()
        pix_y = torch.round((cam_coords[..., 1] * fy / cam_coords[..., 2]) + cy).long()
        # eliminate pixels outside view frustum
        valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
        valid_vox_coords = self.vox_coords[valid_pix]
        depth_val = depth[pix_y[valid_pix], pix_x[valid_pix]]
        # integrate sdf
        depth_diff = depth_val - pix_z[valid_pix]
        # all points 1. inside frustum 2. with valid depth 3. outside -truncate_dist
        dist = torch.clamp(depth_diff / self.sdf_trunc, max=1)
        valid_pts = (depth_val > 0.) & (depth_diff >= -self.sdf_trunc)
        neg_pts = (depth_val > 0.) & (depth_diff < -self.sdf_trunc)
        neg_vox_coords = valid_vox_coords[neg_pts]
        self.aux_vol[neg_vox_coords.T[0], neg_vox_coords.T[1], neg_vox_coords.T[2]] = True
        valid_vox_coords = valid_vox_coords[valid_pts]
        valid_dist = dist[valid_pts]
        assert dist_func in ['point2point', 'point2plane']
        if dist_func == 'point2plane':
            n = self.normal_estimation(depth, cam_intr)
            valid_cam_coords = cam_coords[..., 0:3][valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
            rays = valid_cam_coords / torch.linalg.norm(valid_cam_coords, dim=1, keepdim=True)
            valid_n = n[pix_y[valid_pix], pix_x[valid_pix]][valid_pts]
            cos = torch.abs(torch.sum(rays * valid_n, dim=1))
            valid_dist = valid_dist * cos
        w_old = self.w_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        sdf_old = self.sdf_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        weight = 1.0
        w_new = w_old + weight
        sdf_new = (w_old * sdf_old + weight * valid_dist) / w_new
        self.sdf_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]] = sdf_new
        if self.fuse_color and rgb is not None:
            rgb = self.to_tensor(rgb)
            rgb_old = self.rgb_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
            rgb = rgb[pix_y[valid_pix], pix_x[valid_pix]]
            valid_rgb = rgb[valid_pts]
            rgb_new = (w_old[:, None] * rgb_old + weight * valid_rgb) / w_new[:, None]
            self.rgb_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2], :] = rgb_new
        w_new = torch.clamp(w_new, max=20)
        self.w_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]] = w_new

    @property
    def shape(self):
        return self.sdf_vol.shape

    @property
    def post_processed_vol(self):
        sdf_vol = self.sdf_vol.clone()
        in_obj_ids = torch.logical_and(self.w_vol == 0, self.aux_vol)
        sdf_vol[in_obj_ids] = -1.0
        return sdf_vol

    def reset(self):
        self.sdf_vol = torch.ones(*self.res).to(self.dt).to(self.dev)
        self.w_vol = torch.zeros(*self.res).to(self.dt).to(self.dev)
        self.aux_vol = torch.zeros(*self.res).to(torch.bool).to(self.dev)
        if self.fuse_color:
            self.rgb_vol = torch.zeros(*self.res, 3).to(self.dt).to(self.dev)

    def to_tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.dt).to(self.dev)

    def marching_cubes(self, step_size=1, level=0, use_post_processed=True, smooth=True):
        b = time.time()
        if use_post_processed:
            sdf_vol = self.post_processed_vol
        else:
            sdf_vol = self.sdf_vol
        if smooth:
            sdf_vol = self.gaussian_smooth(sdf_vol)
        sdf_vol = sdf_vol.cpu().numpy()
        verts, faces, norms, vals = measure.marching_cubes(sdf_vol, level=level, step_size=step_size)
        norms = -norms
        verts_ids = np.round(verts).astype(int)
        verts = verts * self.vox_len.cpu().numpy() + self.origin.cpu().numpy().reshape(1, 3)
        if self.fuse_color:
            rgbs = self.rgb_vol.cpu().numpy()[verts_ids[:, 0], verts_ids[:, 1], verts_ids[:, 2]].astype(np.uint8)
        else:
            rgbs = np.ones_like(verts).astype(np.uint8) * 255
        e = time.time()
        print('elapse time on marching cubes: {:04f}ms'.format((e - b)*1000))
        return verts, faces, norms, rgbs

    def compute_pcl(self, threshold=0.2, use_post_processed=True, smooth=True):
        b = time.time()
        if use_post_processed:
            sdf_vol = self.post_processed_vol
        else:
            sdf_vol = self.sdf_vol
        if smooth:
            sdf_vol = self.gaussian_smooth(sdf_vol)
        valid_vox = torch.abs(sdf_vol) < threshold
        xyz = self.vox_coords[valid_vox] * self.vox_len + self.origin.reshape(1, 3)
        if self.fuse_color:
            rgb = self.rgb_vol[valid_vox]
        else:
            rgb = torch.ones_like(xyz) * 255
        e = time.time()
        print('elapse time on extracting point cloud: {:04f}ms'.format((e - b) * 1000))
        return xyz, rgb.to(torch.uint8)

    def sdf_info(self):
        print('origin: ')
        print(self.origin)
        print('resolution')
        print(self.res)
        print('volume bounds: ')
        print('x_min: {:04f}m'.format(self.origin[0]))
        print('x_max: {:04f}m'.format(self.origin[0]+self.res[0]*self.vox_len))
        print('y_min: {:04f}m'.format(self.origin[1]))
        print('y_max: {:04f}m'.format(self.origin[1]+self.res[1]*self.vox_len))
        print('z_min: {:04f}m'.format(self.origin[2]))
        print('z_max: {:04f}m'.format(self.origin[2]+self.res[2]*self.vox_len))
        print('voxel length: {:06f}m'.format(self.vox_len))

    def depth2cloud(self, depth, intrinsic):
        h, w = depth.shape
        xm, ym = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        pixel_coords = torch.stack([xm, ym, torch.ones_like(xm)], dim=-1).to(self.dt).to(self.dev)  # H x W x 3
        inv_intrinsic = torch.linalg.inv(intrinsic)
        pcl = depth.unsqueeze(dim=-1) * (pixel_coords @ inv_intrinsic.T)  # ordered point cloud map
        return pcl

    def normal_estimation(self, depth, intrinsic, kernel_size=5, normalized=True):
        """
        Estimate surface normals in camera frame given the depth map and the intrinsic matrix of the camera.
        We estimate the surface normal by approximating the surface z = f(x, y) as a linear function of x and y.
        For more details, please refer to "A Fast Method For Computing Principal Curvatures From Range Images".
        :param depth: (torch.Tensor) [H, W] the depth map.
        :param intrinsic: (torch.Tensor) [3, 3] the intrinsic matrix of the camera.
        :param kernel_size: (int) the size of the kernel that containing the pixels used to estimate surface normal.
        :param normalized: (bool) whether normalize the surface normal or not.
        :return: (torch.Tensor) [H, W, 3] the surface normal map in camera frame.
        """
        one_hot_w = [k.reshape(kernel_size, kernel_size) for k in torch.eye(kernel_size*kernel_size).unbind(dim=-1)]
        one_hot_w = torch.stack(one_hot_w, dim=0).unsqueeze(dim=1).to(self.dt).to(self.dev)  # ks*ks x 1 x ks x ks
        w1 = torch.ones((1, 1, kernel_size, kernel_size), device=self.dev, dtype=self.dt)
        pixel_coords = self.depth2cloud(depth, intrinsic).permute(2, 0, 1).unsqueeze(dim=0)
        num_valid = conv2d((depth.unsqueeze(dim=0).unsqueeze(dim=0) > 0).to(self.dt), w1, padding='same')
        pixel_coords_sum = [conv2d(coords, w1, padding='same') for coords in pixel_coords.unsqueeze(dim=0).unbind(dim=2)]
        pixel_coords_sum = torch.cat(pixel_coords_sum, dim=1)
        mean = pixel_coords_sum / num_valid  # 1 x 3 x H x W
        xyz = [conv2d(coords, one_hot_w, padding='same') for coords in pixel_coords.unsqueeze(dim=0).unbind(dim=2)]
        xyz = torch.stack(xyz, dim=2).squeeze(dim=0)  # ks^2 x 3 x H x W
        x_cen, y_cen, z_cen = (xyz - mean).unbind(dim=1)  # ks^2 x H x W
        m00 = torch.sum(x_cen * x_cen, dim=0)
        m01 = torch.sum(x_cen * y_cen, dim=0)
        m10 = m01
        m11 = torch.sum(y_cen * y_cen, dim=0)
        v0 = torch.sum(x_cen * z_cen, dim=0)
        v1 = torch.sum(y_cen * z_cen, dim=0)
        det = m00 * m11 - m01 * m10
        a, b = (m11 * v0 - m01 * v1) / det, (-m10 * v0 + m00 * v1) / det
        # todo: not sure if they are valid ops
        a[torch.isnan(a)], b[torch.isnan(b)] = 0, 0
        a[torch.isinf(a)], b[torch.isinf(b)] = 1000, 1000
        if normalized:
            norm = torch.sqrt(1 + a * a + b * b)
            n = torch.stack([a / norm, b / norm, -1 / norm], dim=-1)
        else:
            n = torch.stack([a, b, -torch.ones_like(a, device=self.dev, dtype=self.dt)], dim=-1)
        return n

    def gaussian_smooth(self, volume, kernel_size=3, sigma_square=0.7):
        if len(volume.shape) != 5:
            volume = torch.unsqueeze(torch.unsqueeze(volume, dim=0), dim=0)
        h = kernel_size // 2
        kernel_ids = torch.arange(kernel_size, device=self.dev, dtype=self.dt)
        x, y, z = torch.meshgrid(kernel_ids, kernel_ids, kernel_ids, indexing='ij')
        kernel = torch.exp(-((x - h) ** 2 + (y - h) ** 2 + (z - h) ** 2) / (2 * sigma_square))
        kernel /= torch.sum(kernel)
        kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=0)
        volume = conv3d(volume, weight=kernel, stride=1, padding=h)
        volume = torch.squeeze(volume)
        return volume

    def interpolation(self, coords,
                      sdf=True, rgb=False, weight=False, grad=False,
                      use_post_processed=True, smooth=True,
                      mode='bilinear', padding_mode='border'):
        ids = self.get_ids(coords)
        ids = ids.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=0)  # B*N*1*1*3
        ids = torch.stack([ids[..., 2], ids[..., 1], ids[..., 0]], dim=-1)
        h, w, d = self.res
        size = torch.tensor([d - 1, w - 1, h - 1]).to(self.dt).to(self.dev).reshape((1, 1, 1, 1, 3))
        grid = ids / size * 2 - 1  # [-1, 1]
        out = []
        if sdf:
            sdf_vol = self.post_processed_vol if use_post_processed else self.sdf_vol
            if smooth:
                sdf_vol = self.gaussian_smooth(sdf_vol)
            sdf_vol = sdf_vol.view(1, 1, h, w, d)
            out.append(grid_sample(sdf_vol, grid, mode=mode, padding_mode=padding_mode, align_corners=True).squeeze())
        if rgb:
            rgb_vol = self.rgb_vol.permute(3, 0, 1, 2).unsqueeze(dim=0)
            out.append(grid_sample(rgb_vol, grid, mode='nearest', padding_mode=padding_mode, align_corners=True).squeeze())
        if weight:
            w_vol = self.w_vol.view(1, 1, h, w, d)
            out.append(grid_sample(w_vol, grid, mode=mode, padding_mode=padding_mode, align_corners=True).squeeze())
        if grad and self.grad_vol is not None:
            grad_vol = self.grad_vol.permute(3, 0, 1, 2).unsqueeze(dim=0)
            out.append(grid_sample(grad_vol, grid, mode='nearest', padding_mode=padding_mode, align_corners=True).squeeze())
        return out if len(out) > 1 else out[0]

    def get_ids(self, coords):
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        coords = coords.to(self.dt).to(self.dev)
        ids = (coords - self.origin.view(1, 3)) / self.vox_len
        return ids

    def get_coords(self, ids):
        if isinstance(ids, np.ndarray):
            ids = torch.from_numpy(ids)
        ids = ids.to(self.dt).to(self.dev)
        coords = ids * self.vox_len + self.origin.view(1, 3)
        return coords

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
    def write_pcl(filename, xyz, rgb, n=None):
        """
        Save a point cloud to a polygon .ply file.
        :param filename: The file name of the.ply file.
        :param xyz: (torch.Tensor.float) [N, 3] the point cloud
        :param rgb: (torch.Tensor.uint8) [N, 3] the corresponding rgb values of the point cloud
        :param n: (torch.Tensor.float) [N, 3] optional, the corresponding vertex normals
        :return: None
        """
        xyz = xyz.cpu().numpy()
        rgb = rgb.cpu().numpy()
        n = n.cpu().numpy() if n is not None else None
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (xyz.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        if n is not None:
            ply_file.write("property float nx\n")
            ply_file.write("property float ny\n")
            ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        # Write vertex list
        if n is None:
            for i in range(xyz.shape[0]):
                ply_file.write("%f %f %f %d %d %d\n" % (
                    xyz[i, 0], xyz[i, 1], xyz[i, 2],
                    rgb[i, 0], rgb[i, 1], rgb[i, 2],))
        else:
            for i in range(xyz.shape[0]):
                ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
                    xyz[i, 0], xyz[i, 1], xyz[i, 2],
                    n[i, 0], n[i, 1], n[i, 2],
                    rgb[i, 0], rgb[i, 1], rgb[i, 2],))


class PSDF(TSDF):
    def __init__(self, origin, resolution, voxel_length, truncated_length=0.005,
                 fuse_color=False, device='cuda', dtype=torch.float):
        super().__init__(origin, resolution, voxel_length, truncated_length, fuse_color, device, dtype)
        self.w_vol = torch.ones(*self.res).to(self.dt).to(self.dev) * 10  # sigma2 volume

    def reset(self):
        super().reset()
        self.w_vol = torch.ones(*self.res).to(self.dt).to(self.dev) * 10

    def integrate(self, depth, intrinsic, camera_pose, dist_func='point2point', rgb=None):
        """
        Integrate RGB-D frame into SDF volume (naive SDF)
        :param depth: (ndarray.float) [H, W] a depth map whose unit is meter.
        :param intrinsic: (ndarray.float) [3, 3] the camera intrinsic matrix
        :param camera_pose: (ndarray.float) [4, 4] the transformation from world to camera frame
        :param dist_func: (str) point2point or point2plane
        :param rgb: (ndarray.uint8) [H, W, 3] a color image
        :return: None
        """
        depth = self.to_tensor(depth)
        cam_intr = self.to_tensor(intrinsic)
        n = self.normal_estimation(depth, cam_intr)
        fx, fy, cx, cy = cam_intr[0, 0], cam_intr[1, 1], cam_intr[0, 2], cam_intr[1, 2]
        cam_pose = self.to_tensor(camera_pose)  # w2c
        im_h, im_w = depth.shape
        c2w = torch.inverse(cam_pose)
        cam_coords = self.world_coords @ c2w.T  # world coordinates represented in camera frame
        pix_z = cam_coords[..., 2]
        # project all the voxels back to image plane
        pix_x = torch.round((cam_coords[..., 0] * fx / cam_coords[..., 2]) + cx).long()
        pix_y = torch.round((cam_coords[..., 1] * fy / cam_coords[..., 2]) + cy).long()
        # eliminate pixels outside view frustum
        valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
        valid_vox_coords = self.vox_coords[valid_pix]
        depth_val = depth[pix_y[valid_pix], pix_x[valid_pix]]
        # integrate sdf
        depth_diff = depth_val - pix_z[valid_pix]
        # all points 1. inside frustum 2. with valid depth 3. outside -truncate_dist
        dist = torch.clamp(depth_diff / self.sdf_trunc, max=1)
        valid_pts = (depth_val > 0.) & (depth_diff >= -self.sdf_trunc)
        neg_pts = (depth_val > 0.) & (depth_diff < -self.sdf_trunc)
        neg_vox_coords = valid_vox_coords[neg_pts]
        self.aux_vol[neg_vox_coords.T[0], neg_vox_coords.T[1], neg_vox_coords.T[2]] = True
        valid_vox_coords = valid_vox_coords[valid_pts]
        valid_dist = dist[valid_pts]
        assert dist_func in ['point2point', 'point2plane']
        if dist_func == 'point2plane':
            valid_cam_coords = cam_coords[..., 0:3][valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
            rays = valid_cam_coords / torch.linalg.norm(valid_cam_coords, dim=1, keepdim=True)
            valid_n = n[pix_y[valid_pix], pix_x[valid_pix]][valid_pts]
            cos = torch.abs(torch.sum(rays * valid_n, dim=1))
            valid_dist = valid_dist * cos
        valid_depth_val = depth_val[valid_pts]
        n = n[pix_y[valid_pix], pix_x[valid_pix]]
        valid_n = n[valid_pts]
        valid_cos_theta = torch.acos(torch.abs(valid_n[:, 2]))
        sigma = 0.0012 + 0.0019 * (valid_depth_val - 0.4) ** 2 + 0.0001 / torch.sqrt(valid_depth_val) * valid_cos_theta ** 2 / ((torch.pi / 2 - valid_cos_theta) ** 2 + 1e-7)
        sigma2 = (sigma / self.sdf_trunc) ** 2  # + (torch.exp(0.3 * torch.abs(valid_dist)) - 1)
        sigma2_old = self.w_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        sdf_old = self.sdf_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        sigma2_sum = sigma2_old + sigma2
        sdf_new = (sigma2_old * valid_dist + sigma2 * sdf_old) / sigma2_sum
        self.sdf_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]] = sdf_new
        if self.fuse_color and rgb is not None:
            rgb = self.to_tensor(rgb)
            rgb_old = self.rgb_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
            rgb = rgb[pix_y[valid_pix], pix_x[valid_pix]]
            valid_rgb = rgb[valid_pts]
            rgb_new = (sigma2_old[:, None] * valid_rgb + sigma2[:, None] * rgb_old) / sigma2_sum[:, None]
            self.rgb_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2], :] = rgb_new
        self.w_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]] = (sigma2_old * sigma2) / sigma2_sum

    @property
    def post_processed_vol(self):
        sdf_vol = self.sdf_vol.clone()
        in_obj_ids = torch.logical_and(self.w_vol == 10, self.aux_vol)
        sdf_vol[in_obj_ids] = -1.0
        return sdf_vol


class GradientSDF(TSDF):
    def __init__(self, origin, resolution, voxel_length, truncated_length=0.005,
                 fuse_color=False, device='cuda', dtype=torch.float):
        super().__init__(origin, resolution, voxel_length, truncated_length, fuse_color, device, dtype)
        self.grad_vol = torch.zeros(*self.res, 3).to(self.dt).to(self.dev)  # scaled gradient

    def reset(self):
        super().reset()
        self.grad_vol = torch.zeros(*self.res, 3).to(self.dt).to(self.dev)

    def integrate(self, depth, intrinsic, camera_pose, dist_func='point2point', rgb=None):
        """
        Integrate RGB-D frame into SDF volume (naive SDF)
        :param depth: (ndarray.float) [H, W] a depth map whose unit is meter.
        :param intrinsic: (ndarray.float) [3, 3] the camera intrinsic matrix
        :param camera_pose: (ndarray.float) [4, 4] the transformation from world to camera frame
        :param dist_func: (str) point2point or point2plane
        :param rgb: (ndarray.uint8) [H, W, 3] a color image
        :return: None
        """
        depth = self.to_tensor(depth)
        cam_intr = self.to_tensor(intrinsic)
        n = self.normal_estimation(depth, cam_intr)
        fx, fy, cx, cy = cam_intr[0, 0], cam_intr[1, 1], cam_intr[0, 2], cam_intr[1, 2]
        cam_pose = self.to_tensor(camera_pose)  # w2c
        im_h, im_w = depth.shape
        c2w = torch.inverse(cam_pose)
        cam_coords = self.world_coords @ c2w.T  # world coordinates represented in camera frame
        pix_z = cam_coords[..., 2]
        # project all the voxels back to image plane
        pix_x = torch.round((cam_coords[..., 0] * fx / cam_coords[..., 2]) + cx).long()
        pix_y = torch.round((cam_coords[..., 1] * fy / cam_coords[..., 2]) + cy).long()
        # eliminate pixels outside view frustum
        valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
        valid_vox_coords = self.vox_coords[valid_pix]
        depth_val = depth[pix_y[valid_pix], pix_x[valid_pix]]
        # integrate sdf
        depth_diff = depth_val - pix_z[valid_pix]
        # all points 1. inside frustum 2. with valid depth 3. outside -truncate_dist
        dist = torch.clamp(depth_diff / self.sdf_trunc, max=1)
        valid_pts = (depth_val > 0.) & (depth_diff >= -self.sdf_trunc)
        neg_pts = (depth_val > 0.) & (depth_diff < -self.sdf_trunc)
        neg_vox_coords = valid_vox_coords[neg_pts]
        self.aux_vol[neg_vox_coords.T[0], neg_vox_coords.T[1], neg_vox_coords.T[2]] = True
        valid_vox_coords = valid_vox_coords[valid_pts]
        valid_dist = dist[valid_pts]
        assert dist_func in ['point2point', 'point2plane']
        if dist_func == 'point2plane':
            valid_cam_coords = cam_coords[..., 0:3][valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
            rays = valid_cam_coords / torch.linalg.norm(valid_cam_coords, dim=1, keepdim=True)
            valid_n = n[pix_y[valid_pix], pix_x[valid_pix]][valid_pts]
            cos = torch.abs(torch.sum(rays * valid_n, dim=1))
            valid_dist = valid_dist * cos
        w_old = self.w_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        sdf_old = self.sdf_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        weight = 1.0
        w_new = w_old + weight
        sdf_new = (w_old * sdf_old + weight * valid_dist) / w_new
        self.sdf_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]] = sdf_new
        g_old = self.grad_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        valid_n_c = n[pix_y[valid_pix], pix_x[valid_pix]][valid_pts]
        valid_n_w = valid_n_c @ cam_pose[0:3, 0:3].T
        g_new = (w_old[:, None] * g_old + weight * valid_n_w) / w_new[:, None]
        self.grad_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2], :] = g_new
        if self.fuse_color and rgb is not None:
            rgb = self.to_tensor(rgb)
            rgb_old = self.rgb_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
            rgb = rgb[pix_y[valid_pix], pix_x[valid_pix]]
            valid_rgb = rgb[valid_pts]
            rgb_new = (w_old[:, None] * rgb_old + weight * valid_rgb) / w_new[:, None]
            self.rgb_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2], :] = rgb_new
        # w_new = torch.clamp(w_new, max=20)
        self.w_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]] = w_new

    def compute_pcl(self, threshold=0.2, use_post_processed=True, smooth=True):
        b = time.time()
        if use_post_processed:
            sdf_vol = self.post_processed_vol
        else:
            sdf_vol = self.sdf_vol
        if smooth:
            sdf_vol = self.gaussian_smooth(sdf_vol)
        valid_vox = torch.abs(sdf_vol) < threshold
        xyz = self.vox_coords[valid_vox] * self.vox_len + self.origin.reshape(1, 3)
        if self.fuse_color:
            rgb = self.rgb_vol[valid_vox]
        else:
            rgb = torch.ones_like(xyz) * 255
        g = self.grad_vol[valid_vox]
        n = g / torch.linalg.norm(g, dim=1, keepdim=True)
        e = time.time()
        print('elapse time on extracting point cloud: {:04f}ms'.format((e - b) * 1000))
        return xyz, rgb.to(torch.uint8), n
