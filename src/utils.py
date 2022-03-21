from torch.nn.functional import grid_sample
import torch


class ImageProcessor(object):
    @staticmethod
    def interpolate(img, ids, mode='bilinear', padding_mode='border'):
        """
        retrieve image values given pixel indices.
        :param img: An HxWx(C)-D torch tensor representing the image.
        :param ids: An Nx2-D torch tensor representing the pixel indices.
        :param mode: 'bilinear' | 'nearest' | 'bicubic'
        :param padding_mode: 'zeros' | 'border' | 'reflection'
        :return: An Nx(C)-D torch tensor representing the retrieved values.
        """
        dev = img.device
        dtype = img.dtype
        h, w = img.shape[0:2]
        multi_channel = len(img.shape) == 3
        size = torch.tensor([w-1, h-1], dtype=dtype, device=dev).reshape(1, 1, 1, 2)
        if multi_channel:
            img = img.permute((2, 0, 1)).unsqueeze(dim=0)  # 1 x C x H x W
        else:
            img = img.unsqueeze(dim=0).unsqueeze(dim=0)  # 1 x 1 x H x W
        ids = ids.unsqueeze(dim=0).unsqueeze(dim=0)  # 1 x 1 x N x 2
        ids = torch.stack([ids[..., 1], ids[..., 0]], dim=-1)
        grid = ids / size * 2 - 1
        out = grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)  # 1 x C x 1 x N
        out = out.squeeze(dim=2).squeeze(dim=0)  # C x N
        out = out.T if multi_channel else out.squeeze(dim=0)
        return out

    @staticmethod
    def pixel2point(depth, intrinsic):
        """
        The projection of depth image pixels into 3D points in camera frame.
        :param depth: An HxW-D torch tensor representing the depth map.
        :param intrinsic: A 3x3-D torch tensor representing the intrinsic matrix of the camera.
        :return: An HxWx3-D torch tensor representing the pixel positions in camera frame.
        """
        dev = depth.device
        dtype = depth.dtype
        h, w = depth.shape
        hm, wm = torch.meshgrid(torch.arange(h), torch.arange(w))
        pixel_coords = torch.stack([wm, hm, torch.ones_like(hm)], dim=-1).to(dtype).to(dev)  # H x W x 3
        inv_intrinsic = torch.linalg.inv(intrinsic)
        pixel_pos = depth.unsqueeze(dim=-1) * (pixel_coords @ inv_intrinsic.T)
        return pixel_pos

    @staticmethod
    def bilateral_filter(depth, kernel=7, sigma_d=7/3, sigma_r=0.01/3):
        """
        https://en.wikipedia.org/wiki/Bilateral_filter
        :param depth: depth: An HxW-D torch tensor representing the depth map.
        :param kernel: kernel: the size of the kernel that containing the pixels used to re-estimate the depth value.
        :param sigma_d: spatial smoothing parameter.
        :param sigma_r: range smoothing parameter.
        :return: denoised depth map.
        """
        assert kernel % 2 == 1  # make sure the patch_size is odd.
        dev = depth.device
        dtype = depth.dtype
        h, w = depth.shape
        shift_ids = torch.cat([ids.reshape(-1, 1) for ids in torch.meshgrid(torch.arange(kernel), torch.arange(kernel))], dim=1) - kernel // 2
        num_shift = shift_ids.shape[0]
        weighted_depth = torch.zeros_like(depth, dtype=dtype, device=dev)
        w_sum = torch.zeros_like(depth, dtype=dtype, device=dev)
        for i in range(num_shift):
            h0c, w0c = max(0, 0 + shift_ids[i][0]), max(0, 0 + shift_ids[i][1])
            h1c, w1c = min(h, h + shift_ids[i][0]), min(w, w + shift_ids[i][1])
            h0a, w0a = max(0, 0 - shift_ids[i][0]), max(0, 0 - shift_ids[i][1])
            h1a, w1a = min(h, h - shift_ids[i][0]), min(w, w - shift_ids[i][1])
            shift_dist = shift_ids[i] @ shift_ids[i] / (2 * sigma_d ** 2)
            depth_dist = (depth[h0a:h1a, w0a:w1a] - depth[h0c:h1c, w0c:w1c]) ** 2 / (2 * sigma_r ** 2)
            curr_w = torch.exp(-shift_dist-depth_dist)
            weighted_depth[h0a:h1a, w0a:w1a] += curr_w * depth[h0a:h1a, w0a:w1a]
            w_sum[h0a:h1a, w0a:w1a] += curr_w
        norm_depth = weighted_depth / w_sum
        return norm_depth

    def normal_estimation(self, depth, intrinsic, kernel=7):
        """
        Estimate surface normals in camera frame given the depth map and the intrinsic matrix of the camera.
        We estimate the surface normal by approximating the surface z = f(x, y) as a linear function of x and y.
        For more details, please refer to "A Fast Method For Computing Principal Curvatures From Range Images".
        :param depth: An HxW-D torch tensor representing the depth map.
        :param intrinsic: A 3x3-D torch tensor representing the intrinsic matrix of the camera.
        :param kernel: the size of the kernel that containing the pixels used to estimate surface normal.
        :return: An HxWx3-D torch tensor representing the surface normal map in camera frame.
        """
        assert kernel % 2 == 1  # make sure the patch_size is odd.
        dev = depth.device
        dtype = depth.dtype
        h, w = depth.shape
        pixel_pos = self.pixel2point(depth, intrinsic)
        shift_ids = torch.cat([ids.reshape(-1, 1) for ids in torch.meshgrid(torch.arange(kernel), torch.arange(kernel))], dim=1) - kernel // 2
        num_shift = shift_ids.shape[0]
        x, y, z = torch.zeros((h, w, num_shift, 3), dtype=dtype, device=dev).unbind(dim=-1)
        num_pixel = torch.zeros_like(depth, dtype=dtype, device=dev)  # # of non-zero pixels in a kernel
        for i in range(num_shift):
            h0c, w0c = max(0, 0+shift_ids[i][0]), max(0, 0+shift_ids[i][1])
            h1c, w1c = min(h, h+shift_ids[i][0]), min(w, w+shift_ids[i][1])
            h0a, w0a = max(0, 0-shift_ids[i][0]), max(0, 0-shift_ids[i][1])
            h1a, w1a = min(h, h-shift_ids[i][0]), min(w, w-shift_ids[i][1])
            x[h0c:h1c, w0c:w1c, i], y[h0c:h1c, w0c:w1c, i], z[h0c:h1c, w0c:w1c, i] = torch.unbind(pixel_pos[h0a:h1a, w0a:w1a],
                                                                                                  dim=-1)
            num_pixel[h0c:h1c, w0c:w1c] += 1
        zero_flag = x == 0
        num_pixel = num_pixel.unsqueeze(dim=-1)  # H x W x 1
        x_mean = torch.cat([torch.sum(x/num_pixel, dim=-1, keepdim=True)]*num_shift, dim=-1)
        y_mean = torch.cat([torch.sum(y/num_pixel, dim=-1, keepdim=True)]*num_shift, dim=-1)
        z_mean = torch.cat([torch.sum(z/num_pixel, dim=-1, keepdim=True)]*num_shift, dim=-1)
        x[zero_flag], y[zero_flag], z[zero_flag] = x_mean[zero_flag], y_mean[zero_flag], z_mean[zero_flag]
        x_cen, y_cen, z_cen = x - x_mean, y - y_mean, z - z_mean
        m00 = torch.sum(x_cen*x_cen, dim=-1)
        m01 = torch.sum(x_cen*y_cen, dim=-1)
        m10 = m01
        m11 = torch.sum(y_cen*y_cen, dim=-1)
        v0 = torch.sum(x_cen*z_cen, dim=-1)
        v1 = torch.sum(y_cen*z_cen, dim=-1)
        det = m00 * m11 - m01 * m10
        a, b = (m11 * v0 - m01 * v1) / det, (-m10 * v0 + m00 * v1) / det
        norm = torch.sqrt(1+a*a+b*b)
        n = torch.stack([a/norm, b/norm, -1/norm], dim=-1)
        return n

