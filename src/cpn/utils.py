from torch.nn.functional import grid_sample
import numpy as np
import torch


def interpolation(volumes, indices, mode='bilinear', padding_mode='zeros'):
    """
    Extract features from volumes using grid_sample function provided by pytorch.
    :param volumes: A B*C*H*W*D float torch tensor.
    :param indices: A B*3*N*1 float torch tensor representing the indices of the volume
    :param mode: interpolation mode to calculate output values 'bilinear' | 'nearest' | 'bicubic'.
    :param padding_mode: padding mode for outside grid values 'zeros' | 'border' | 'reflection'.
    :return: A B*C*N*1 float torch tensor denoting the extracted features.
    """
    _, _, h, w, d = volumes.shape
    size = torch.from_numpy(np.array([d-1, w-1, h-1], dtype=np.float32).reshape((1, 1, 1, 1, 3))).to(indices.device)
    indices = torch.unsqueeze(indices.permute(0, 2, 3, 1), dim=3)  # B*N*1*1*3
    indices = torch.stack([indices[..., 2], indices[..., 1], indices[..., 0]], dim=-1)
    grid = indices / size * 2 - 1   # [-1, 1]
    output = grid_sample(volumes, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    output = torch.squeeze(output, dim=4)
    return output


def extract_sdf(pts, volumes, origin, resolution, mode='bilinear', padding_mode='border'):
    """
    Extract sdf values from volumes according to query points.
    :param pts: A B*3*num_sample_pts*num_pose-D float torch tensor.
    :param volumes: A B*C*H*W*D-d float torch tensor.
    :param origin: A B*3-d float torch tensor denoting the origin of the sdf volume.
    :param resolution: A B-d float torch tensor denoting the resolution of the sdf volume.
    :param mode: 'bilinear' | 'nearest' | 'bicubic'
    :param padding_mode:'zeros' | 'border' | 'reflection'
    :return: A B*1*num_sample_pts*num_pose-d torch tensor representing the sdf vector for each grasp pose.
    """
    b = origin.shape[0]
    ids = (pts - origin.view(b, 3, 1, 1)) / resolution.view(b, 1, 1, 1)
    sdf_vec = interpolation(volumes, ids, mode=mode, padding_mode=padding_mode)
    return sdf_vec


