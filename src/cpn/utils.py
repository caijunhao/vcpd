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


def sample_contact_points(sdf, th_a=30, th_s=0.2, start=0.01, end=0.08, num_step=50,
                          post_processed=True, gaussian_blur=True, step_size=2):
    dtype = sdf.dt
    dev = sdf.dev
    th_a = torch.deg2rad(torch.tensor(th_a, dtype=dtype, device=dev))
    v, _, n, _ = sdf.marching_cubes(step_size=step_size, use_post_processed=post_processed, smooth=gaussian_blur)
    v, n = torch.tensor(v, dtype=dtype, device=dev), torch.tensor(n, dtype=dtype, device=dev)
    flag_s = torch.abs(sdf.interpolation(v, use_post_processed=post_processed, smooth=gaussian_blur)) < th_s
    v, n = v[flag_s], n[flag_s]
    # filter out contact points whose surface normals are outside the angle threshold
    flag_a = torch.abs(n[:, 2]) <= torch.cos(torch.pi / 2 - th_a)
    v, n = v[flag_a], n[flag_a]
    stride = (end - start) / num_step
    samples = n.unsqueeze(dim=1) * (start + torch.arange(num_step, dtype=dtype, device=dev).reshape(1, -1, 1) * stride)
    vs = v.unsqueeze(dim=1) - samples  # N * M * 3
    num_v = vs.shape[0]
    sdv = sdf.interpolation(vs.view(-1, 3),
                            use_post_processed=post_processed, smooth=gaussian_blur).view(num_v, num_step)  # N * M
    min_sdv, min_ids = torch.min(torch.abs(sdv), dim=1)  # N, N
    flag = torch.logical_and(min_sdv < th_s, torch.logical_and(min_ids > 0, min_ids < num_step - 1))
    v, vs, sdv, min_sdv, min_ids = v[flag], vs[flag], sdv[flag], min_sdv[flag], min_ids[flag]  # N'
    num_v = vs.shape[0]
    ids_plus_one, ids_minus_one = min_ids + 1, min_ids - 1
    v_ids = torch.arange(num_v, device=dev)
    flag_sign = (sdv[v_ids, ids_plus_one] - sdv[v_ids, ids_minus_one]) > 0
    v, vs, min_sdv, min_ids = v[flag_sign], vs[flag_sign], min_sdv[flag_sign], min_ids[flag_sign]  # N''
    num_v = vs.shape[0]
    v_ids = torch.arange(num_v, device=dev)
    return v, vs[v_ids, min_ids]


def select_gripper_pose(sdf, pg, score, cp1, cp2, gripper_depth,
                        num_angle=32, max_width=0.08,
                        th_s=0.997, th_c=0.999,
                        check_tray=True,
                        post_processed=True, gaussian_blur=True):
    """
    Select collision-free pose according to the quality scores of the contact points.
    :param sdf: the SDF instance
    :param pg: a python dictionary containing the surface vertices of the gripper
    :param score: an N-D torch tensor representing the grasp qualities of contact point pairs.
    :param cp1: an Nx3-D torch tensor representing the 3D locations of the left contact points.
    :param cp2: an Nx3-D torch tensor representing the 3D locations of the right contact points.
    :param gripper_depth:y[:, 0:2]
    :param num_angle: the number of discretized angles in x-z plane
    :param max_width:
    :param th_s:
    :param th_c:
    :param check_tray:
    :param post_processed:
    :param gaussian_blur:
    :return: 4 torch tensor representing sets of gripper position, rotations, and contact points
    """
    dtype = score.dtype
    dev = score.device
    score = torch.sigmoid(score)
    score, rank = torch.sort(score, descending=True)
    th_s = score[0] * th_s  # select top 0.1% contact points as candidates
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
    z[:, 0] = y[:, 1] / z_norm
    z[:, 1] = -y[:, 0] / z_norm
    x = torch.cross(y, z, dim=1)
    x = x / torch.linalg.norm(x, dim=1, keepdim=True)
    # u, _, _ = torch.linalg.svd(y.view(num_cp, 3, 1))  # the shape of u: num_cp * 3 * 3
    # x = u[..., 1]  # num_cp * 3
    # z = torch.cross(x, y, dim=1)
    rot0 = torch.stack([x, y, z], dim=2)  # num_cp * 3 * 3
    angles = torch.arange(num_angle, dtype=dtype, device=dev) / num_angle * torch.pi  # num_angle
    delta_rots = basic_rots(angles, axis='y')  # num_angle * 3 * 3
    rots = torch.matmul(rot0.unsqueeze(dim=1), delta_rots.unsqueeze(dim=0))  # num_cp * num_angle * 3 * 3
    z_flag = rots[..., -1, -1] > 0
    i, j = torch.meshgrid(torch.arange(num_cp, device=dev), torch.arange(num_angle, device=dev))
    i, j = i[z_flag], j[z_flag]
    rots[i, j, :, 0], rots[i, j, :, 2] = -rots[i, j, :, 0], -rots[i, j, :, 2]
    ys = rots[..., 1].unsqueeze(dim=2)  # num_cp * num_angle * 1 * 3
    n_h, n_l, n_r = pg['hand'].shape[0], pg['left_finger'].shape[0], pg['right_finger'].shape[0]
    vs = torch.from_numpy(np.concatenate([pg['hand'], pg['left_finger'], pg['right_finger']], axis=0)).to(dtype).to(dev)
    num_v = vs.shape[0]
    gripper_pos = pos.reshape(num_cp, 1, 3) - rots[..., 2] * gripper_depth  # num_cp * num_angle * 3
    vs = torch.matmul(rots, vs.permute(1, 0).unsqueeze(0).unsqueeze(0)).permute(0, 1, 3, 2) + gripper_pos.unsqueeze(2)  # num_cp * num_angle * num_v * 3
    vs[..., n_h:n_h+n_l, :] = vs[..., n_h:n_h+n_l, :] - offset * ys  # left finger vertices
    vs[..., n_h+n_l:n_h+n_l+n_r, :] = vs[..., n_h+n_l:n_h+n_l+n_r, :] + offset * ys  # right finger vertices
    sdv = sdf.interpolation(vs.reshape(-1, 3),
                            use_post_processed=post_processed, smooth=gaussian_blur).reshape(num_cp, num_angle, num_v)
    if check_tray:
        vs_ids = sdf.get_ids(vs.reshape(-1, 3)).reshape(num_cp, num_angle, num_v, 3)
        oob_flag = torch.logical_or(torch.logical_or(vs_ids[..., 0] < 0, vs_ids[..., 0] >= sdf.shape[0]),
                                    torch.logical_or(vs_ids[..., 1] < 0, vs_ids[..., 1] >= sdf.shape[1]))
        sdv[oob_flag] = -1
    num_free = torch.sum(sdv > 0.2, dim=-1)  # num_cp * num_angle
    flag_c = num_free > torch.max(num_free) * th_c
    rots = rots[flag_c]
    gripper_pos = gripper_pos[flag_c]
    width = torch.cat([width] * num_angle, dim=1)[flag_c]
    distance = torch.cat([distance] * num_angle, dim=1)[flag_c]
    pos = torch.stack([pos] * num_angle, dim=1)[flag_c]
    num_free = num_free[flag_c]
    col_score = num_free / num_v
    # resort the contact points
    # _, ids = torch.sort(score_n, descending=True)
    # rots = rots[ids]
    # gripper_pos = gripper_pos[ids]
    # width = width[ids]
    # distance = distance[ids]
    # pos = pos[ids]
    grasp_directions = rots[..., 1]
    cp1 = (pos + grasp_directions * distance.reshape(-1, 1) / 2)
    cp2 = (pos - grasp_directions * distance.reshape(-1, 1) / 2)
    return gripper_pos, rots, width, cp1, cp2


def clustering(gripper_pos, rots, width, cp1, cp2, th_d=0.02, th_a=-0.707):
    num_cp = cp1.shape[0]
    dtype = cp1.dtype
    dev = cp1.device
    # if # of cp pairs are less than or equal to 7, then randomly select one and return
    if num_cp <= 7:
        ids = torch.argsort(rots[:, -1, -1])
        idx = ids[0]
        # idx = torch.randint(0, num_cp, size=(1,), device=dev)[0]
        return gripper_pos[idx].cpu().numpy(), rots[idx].cpu().numpy(), width[idx].cpu().numpy(), cp1[idx].cpu().numpy(), cp2[idx].cpu().numpy()
    grasp_center = (cp1 + cp2) / 2
    init_num_cls = 7
    cls_flag = torch.zeros(num_cp, dtype=bool, device=dev)
    # randomly sample a center as initial cluster
    init_idx = torch.randint(0, num_cp, size=(1,), device=dev)[0]
    cls_flag[init_idx] = True
    ids = torch.arange(num_cp, device=dev)
    for _ in range(init_num_cls):
        curr_cls = grasp_center[cls_flag].reshape(1, 3) if torch.sum(cls_flag) == 1 else grasp_center[cls_flag]
        dist = torch.linalg.norm(curr_cls.reshape(-1, 1, 3) - grasp_center.reshape(1, -1, 3), dim=-1)
        # farthest center selection
        cls_flag[torch.argmax(torch.sum(dist, dim=0))] = True
    assigned_flag = cls_flag.detach().clone()
    while torch.sum(assigned_flag) < num_cp:
        curr_cls = grasp_center[cls_flag]  # num_cls * 3
        dist = torch.linalg.norm(curr_cls.reshape(-1, 1, 3)-grasp_center.reshape(1, -1, 3), dim=-1)  # num_cls * num_cp
        min_dist, min_ids = torch.min(dist, dim=0)
        assigned_flag = min_dist <= th_d
        unassigned_ids = ids[torch.logical_not(assigned_flag)]
        if unassigned_ids.shape[0] != 0:
            cls_flag[unassigned_ids[0]] = True
            # cls_flag[unassigned_ids[torch.randperm(unassigned_ids.shape[0], device=dev)[0]]] = True
    num_cls = torch.sum(cls_flag)
    num_max = 0
    flag = None
    for i in range(num_cls):
        curr_flag = min_ids == i  # current cluster flag
        curr_num_center = torch.sum(curr_flag)
        if curr_num_center > num_max:
            num_max = curr_num_center
            flag = curr_flag
    curr_centers = grasp_center[flag]
    cls_center = torch.mean(curr_centers, dim=0)
    dist = torch.linalg.norm(curr_centers - cls_center.view(1, 3), dim=1)
    ids = torch.argsort(dist)
    gripper_pos, rots, width, cp1, cp2 = gripper_pos[flag][ids], rots[flag][ids], width[flag][ids], cp1[flag][ids], cp2[flag][ids]
    flag_z = rots[:, -1, -1] <= torch.max(torch.min(rots[:, -1, -1]), torch.tensor(th_a, dtype=dtype, device=dev))
    gripper_pos, rots, width, cp1, cp2 = gripper_pos[flag_z][0], rots[flag_z][0], width[flag_z][0], cp1[flag_z][0], cp2[flag_z][0]
    return gripper_pos.cpu().numpy(), rots.cpu().numpy(), width.cpu().numpy(), cp1.cpu().numpy(), cp2.cpu().numpy()


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
