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


def sample_contact_points(tsdf, th_a=30, th_s=0.2, start=0.01, end=0.06, num_step=50):
    dtype = tsdf.dtype
    dev = tsdf.dev
    th_a = np.deg2rad(th_a)
    v, _, n, _ = tsdf.compute_mesh(step_size=3)
    # filter out contact points whose surface normals are outside the angle threshold
    flag_a = np.abs(n[:, 2]) <= np.cos(np.pi / 2 - th_a)
    v, n = v[flag_a], n[flag_a]
    v, n = torch.tensor(v, dtype=dtype, device=dev), torch.tensor(n, dtype=dtype, device=dev)
    stride = (end - start) / num_step
    samples = n.unsqueeze(dim=1) * (start + torch.arange(num_step, dtype=dtype, device=dev).reshape(1, -1, 1) * stride)
    vs = v.unsqueeze(dim=1) - samples  # N * M * 3
    num_v = vs.shape[0]
    sdv = tsdf.extract_sdf(vs.view(-1, 3)).view(num_v, num_step)  # N * M
    min_sdv, min_ids = torch.min(torch.abs(sdv), dim=1)  # N, N
    flag = torch.logical_and(min_sdv < th_s, torch.logical_and(min_ids > 0, min_ids < num_step - 1))
    v, vs, sdv, min_sdv, min_ids = v[flag], vs[flag], sdv[flag], min_sdv[flag], min_ids[flag]  # N'
    num_v = vs.shape[0]
    ids_plus_one, ids_minus_one = min_ids + 1, min_ids - 1
    v_ids = torch.arange(num_v, device=dev)
    flag_sign = sdv[v_ids, ids_plus_one] - sdv[v_ids, ids_minus_one] > 0
    v, vs, min_sdv, min_ids = v[flag_sign], vs[flag_sign], min_sdv[flag_sign], min_ids[flag_sign]  # N''
    num_v = vs.shape[0]
    v_ids = torch.arange(num_v, device=dev)
    return v, vs[v_ids, min_ids]


def select_gripper_pose(tsdf, pg, score, cp1, cp2, gripper_depth, num_angle=32, max_width=0.08):
    """
    Select collision-free pose according to the quality scores of the contact points.
    :param tsdf: the SDF instance
    :param pg: a python dictionary containing the surface vertices of the gripper
    :param score: an N-D torch tensor representing the grasp qualities of contact point pairs.
    :param cp1: an Nx3-D torch tensor representing the 3D locations of the left contact points.
    :param cp2: an Nx3-D torch tensor representing the 3D locations of the right contact points.
    :param gripper_depth:y[:, 0:2]
    :param num_angle: the number of discretized angles in x-z plane
    :param max_width:
    :return: the 7-DoF gripper pose with the highest grasp quality and free collision.
    """
    dtype = score.dtype
    dev = score.device
    score = torch.sigmoid(score)
    score, rank = torch.sort(score, descending=True)
    th_s = score[0] * 0.97  # select top 3% contact points as candidates
    rank = rank[score >= th_s]
    num_cp = rank.shape[0]
    cp1, cp2 = cp1[rank], cp2[rank]
    pos = (cp1 + cp2) / 2  # num_cp * 3
    y = cp1 - cp2
    distance = torch.linalg.norm(y, dim=1, keepdim=True)
    y = y / distance  # num_cp * 3
    width = torch.clamp(distance + 0.02, 0.0, max_width)  # num_cp * 1
    offset = (max_width - width) / 2
    offset = offset.unsqueeze(-1).unsqueeze(-1)  # num_cp * 1 * 1 * 1
    z = torch.zeros_like(y, dtype=dtype, device=dev)
    z_norm = torch.linalg.norm(y[:, 0:2], dim=1)
    z[:, 0] = torch.abs(y[:, 1]) / z_norm
    z[:, 1] = -torch.abs(y[:, 0]) / z_norm
    x = torch.cross(y, z)
    x = x / torch.linalg.norm(x, dim=1, keepdim=True)
    # u, _, _ = torch.linalg.svd(y.view(num_cp, 3, 1))  # the shape of u: num_cp * 3 * 3
    # x = u[..., 1]  # num_cp * 3
    # z = torch.cross(x, y, dim=1)
    rot0 = torch.stack([x, y, z], dim=2)  # num_cp * 3 * 3
    angles = torch.arange(num_angle, dtype=dtype, device=dev) / num_angle * torch.pi  # num_angle
    delta_rots = basic_rots(angles, axis='y')  # num_angle * 3 * 3
    rots = torch.matmul(rot0.unsqueeze(dim=1), delta_rots.unsqueeze(dim=0))  # num_cp * num_angle * 3 * 3
    z_flag = rots[..., -1, -1] > 0
    rots[z_flag, :, 0], rots[z_flag, :, 2] = -rots[z_flag, :, 0], -rots[z_flag, :, 2]
    ys = rots[..., 1].unsqueeze(dim=2)  # num_cp * num_angle * 1 * 3
    n_h, n_l, n_r = pg['hand'].shape[0], pg['left_finger'].shape[0], pg['right_finger'].shape[0]
    vs = torch.from_numpy(np.concatenate([pg['hand'], pg['left_finger'], pg['right_finger']], axis=0)).to(dtype).to(dev)
    num_v = vs.shape[0]
    gripper_pos = pos.reshape(num_cp, 1, 3) - rots[..., 2] * gripper_depth  # num_cp * num_angle * 3
    vs = torch.matmul(rots, vs.permute(1, 0).unsqueeze(0).unsqueeze(0)).permute(0, 1, 3, 2) + gripper_pos.unsqueeze(2)  # num_cp * num_angle * num_v * 3
    vs[..., n_h:n_h+n_l, :] = vs[..., n_h:n_h+n_l, :] - offset * ys  # left finger vertices
    vs[..., n_h+n_l:n_h+n_l+n_r, :] = vs[..., n_h+n_l:n_h+n_l+n_r, :] + offset * ys  # right finger vertices
    sdv = tsdf.extract_sdf(vs.reshape(-1, 3)).reshape(num_cp, num_angle, num_v)
    num_free = torch.sum(sdv > 0, dim=-1)  # num_cp * num_angle
    flag_s = num_free > torch.max(num_free) * 0.99
    rots = rots[flag_s]
    gripper_pos = gripper_pos[flag_s]
    width = torch.cat([width] * num_angle, dim=1)[flag_s]
    num_free = num_free[flag_s]
    score_n = -rots[:, 2, 2] + num_free / num_v * 20
    _, ids = torch.sort(score_n, descending=True)
    # ids = ids[torch.randperm(ids.shape[0])]
    rots = rots[ids]
    gripper_pos = gripper_pos[ids].squeeze(dim=1)
    width = width[ids]
    # recompute the contact points
    distance = torch.cat([distance] * num_angle, dim=1)[flag_s]
    distance = distance[ids]
    pos = torch.stack([pos] * num_angle, dim=1)[flag_s]
    pos = pos[ids]
    grasp_directions = rots[:, 1]
    cp1 = (pos + grasp_directions * distance.reshape(-1, 1) / 2)
    cp2 = (pos - grasp_directions * distance.reshape(-1, 1) / 2)
    # debug: sample contact points and visualize
    # ids = torch.randperm(cp1.shape[0])[0:min(50, cp1.shape[0])]
    # selected_cp1, selected_cp2 = cp1[ids].cpu().numpy(), cp2[ids].cpu().numpy()
    # import pybullet as p
    # radius = 0.003
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # cp1s = [p.createMultiBody(0,
    #                           p.createCollisionShape(p.GEOM_SPHERE, radius),
    #                           p.createVisualShape(p.GEOM_SPHERE, radius, rgbaColor=[1, 0, 0, 1]),
    #                           basePosition=cp) for cp in selected_cp1]
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # cp2s = [p.createMultiBody(0,
    #                           p.createCollisionShape(p.GEOM_SPHERE, radius),
    #                           p.createVisualShape(p.GEOM_SPHERE, radius, rgbaColor=[0, 1, 0, 1]),
    #                           basePosition=cp) for cp in selected_cp2]
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # lines = [p.addUserDebugLine(selected_cp1[pid], selected_cp2[pid],
    #                             lineColorRGB=np.random.uniform(size=3),
    #                             lineWidth=0.1) for pid in range(selected_cp2.shape[0])]
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # [p.removeBody(cp) for cp in cp1s]
    # [p.removeBody(cp) for cp in cp2s]
    # [p.removeUserDebugItem(line) for line in lines]
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # del p
    # /debug
    return gripper_pos[0].cpu().numpy(), rots[0].cpu().numpy(), width[0].cpu().numpy(), cp1[0].cpu().numpy(), cp2[0].cpu().numpy()


def basic_rots(angles, axis):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    shape = list(cos.shape)
    if axis == 'x':
        rots = torch.stack([one, zero, zero,
                            zero, cos, -sin,
                            zero, sin, cos], dim=1).reshape(shape + [3, 3])
    elif axis == 'y':
        rots = torch.stack([cos, zero, sin,
                            zero, one, zero,
                            -sin, zero, cos], dim=1).reshape(shape + [3, 3])
    elif axis == 'z':
        rots = torch.stack([cos, -sin, zero,
                            sin, cos, zero,
                            zero, zero, one], dim=1).reshape(shape+[3, 3])
    else:
        raise ValueError('invalid axis')
    return rots