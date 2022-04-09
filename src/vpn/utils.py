from scipy.spatial.transform import Rotation as R
from functools import partial
from torch.nn.functional import grid_sample
from itertools import count
import numpy as np
import torch
import heapq


class PriorityQueue(object):
    def __init__(self):
        self._elements = list()
        self.counter = count()  # the counter is added to avoid ValueError caused by numpy array comparison

    def empty(self):
        return not self._elements

    def push(self, item, priority):
        # why counter here:
        # https://stackoverflow.com/questions/39504333/python-heapq-heappush-the-truth-value-of-an-array-with-more-than-one-element-is
        heapq.heappush(self._elements, (priority, next(self.counter), item))

    def pop(self):
        item = heapq.heappop(self._elements)
        return item[0], item[2]  # remove counter

    def get(self):
        item = self._elements[0]
        return item[0], item[2]  # remove counter

    def nlargest(self, n):
        n = min(n, len(self._elements))
        items = heapq.nlargest(n, self._elements)
        return list(map(lambda x: (x[0], x[2]), items))
        # return list(map(lambda x: x[1], heapq.nlargest(n, self._elements)))  # without scores

    def __len__(self):
        return len(self._elements)


def interpolation(volumes, indices):
    """
    Extract features from volumes according to the indices.
    Since the indices may be float indices, we need to interpolate features near to the indices
    to get the exact output features.
    :param volumes: A B*C*H*W*D float torch tensor.
    :param indices: A B*3*N*1 float torch tensor representing the indices of the volume
    :return: A B*C*N*1 float torch tensor denoting the extracted features.
    """
    _, _, h, w, d = volumes.shape
    indices = torch.squeeze(indices.clone(), dim=-1)  # B*3*N
    indices_x, indices_y, indices_z = indices[:, 0, :], indices[:, 1, :], indices[:, 2, :]
    indices_x[indices_x + 1 >= h] = h - 1 - 1e-5
    indices_y[indices_y + 1 >= w] = w - 1 - 1e-5
    indices_z[indices_z + 1 >= d] = d - 1 - 1e-5
    indices = torch.stack([indices_x, indices_y, indices_z], dim=1)
    ids_min = torch.floor(indices).type(torch.LongTensor).to(indices.device)  # B*3*N*1
    # ids_max = torch.LongTensor(torch.ceil(indices))  # B*3*N
    output = 0
    batch_size = volumes.shape[0]
    # w_list = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                offset = torch.tensor([x, y, z], dtype=torch.int64, device=indices.device).reshape(1, 3, 1)
                feature_list = []
                ids = ids_min + offset
                for b in range(batch_size):
                    feature_list.append(volumes[b, :, ids[b, 0, :], ids[b, 1, :], ids[b, 2, :]])  # C*N*1
                features = torch.stack(feature_list, dim=0)
                diff = 1 - torch.abs(indices - ids)
                w = diff[:, 0:1, :] * diff[:, 1:2, :] * diff[:, 2:3, :]  # B * N
                # w_list.append(w)
                output += w * features
    return torch.unsqueeze(output, dim=-1)


def interpolation_v2(volumes, indices, mode='bilinear', padding_mode='zeros'):
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
    sdf_vec = interpolation_v2(volumes, ids, mode=mode, padding_mode=padding_mode)
    return sdf_vec


def euler2rot(euler, seq='zyx'):
    r = R.from_euler(seq, euler)
    return r.as_matrix()


def rank_and_group_poses(sample, scores, device,
                         gripper_points, collision_check=True,
                         angle_threshold=45, score_threshold=0.01, dist_threshold=0.02):
    """
    Select, rank and group gripper poses according to the predicted scores from vpn.
    :param sample: A dictionary consisting of torch tensors used as input of vpn
    :param scores: A B*18*N*1-d torch tensor representing the grasp qualities for all the grasp candidates.
    :param device: The desired device of the tensor.
    :param gripper_points:
    :param collision_check:
    :param angle_threshold:
    :param score_threshold: Threshold to specify the maximum difference between one score and the max score.
    :param dist_threshold: Distance threshold to specify the minimum distance between two groups.
    :param th_z: the minimum z position of grasp candidates
    :return: A dictionary composed of groups of gripper poses with high grasp qualities.
    """
    num_pts = sample['pts'].shape[2]  # shape of pts: B * 3 * N * 1
    pts = torch.unsqueeze(sample['pts'].permute(0, 2, 3, 1), dim=1).repeat(1, 18, 1, 1, 1)
    normals = torch.unsqueeze(sample['normals'].permute(0, 2, 3, 1), dim=1).repeat(1, 18, 1, 1, 1)
    angle_ids = torch.arange(18, dtype=torch.float32).to(device).repeat(1, 1, num_pts, 1).permute(0, 3, 2, 1)
    # remove candidates whose angles between normal and z axis are larger than angle_threshold degree
    normal_flag = normals[..., 2] > np.cos(np.deg2rad(angle_threshold))
    if not torch.any(normal_flag):
        print('no feasible solution found, try again.')
        return None
    scores = scores[normal_flag]  # N1
    pts = pts[normal_flag]  # N1 * 3
    normals = normals[normal_flag]  # N1 * 3
    angle_ids = angle_ids[normal_flag]  # N1
    max_score = torch.max(scores)
    # select grasp candidates with scores larger than max(scores) - score_threshold
    flag = scores > max_score - score_threshold
    scores = scores[flag]  # N2
    pts = pts[flag]  # N2 * 3
    normals = normals[flag]  # N2 * 3
    angle_ids = angle_ids[flag]  # N2
    scores, rank = torch.sort(scores, descending=True)
    # sort grasp candidates according to the scores
    pts = pts[rank]
    normals = normals[rank]
    angle_ids = angle_ids[rank]
    # compute grasp axe according to normals and angle_ids
    grasp_axe_0 = torch.zeros_like(normals, dtype=torch.float32, device=device)
    grasp_axe_0[:, 0], grasp_axe_0[:, 2] = 1, -normals[:, 0] / normals[:, 2]
    grasp_axe_0 = grasp_axe_0 / torch.linalg.norm(grasp_axe_0, dim=1, keepdim=True)
    angles = torch.deg2rad(angle_ids * 10)
    # correct
    cos, sin = torch.cos(angles), torch.sin(angles)
    zeros = torch.zeros_like(cos, dtype=cos.dtype, device=device)
    ones = torch.ones_like(cos, dtype=cos.dtype, device=device)
    rots_n = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=1).reshape(-1, 3, 3)
    y0 = torch.cross(-normals, grasp_axe_0)
    rots_0 = torch.stack([grasp_axe_0, y0, -normals], dim=2)
    rots = torch.matmul(rots_0, rots_n)
    # wrong
    # cos, sin = torch.cos(-angles), torch.sin(-angles)
    # zeros = torch.zeros_like(cos, dtype=cos.dtype, device=device)
    # ones = torch.ones_like(cos, dtype=cos.dtype, device=device)
    # rots_n = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=1).reshape(-1, 3, 3)
    # grasp_axe = torch.bmm(rots_n, grasp_axe_0.view(-1, 3, 1)).view(-1, 3)
    # y = torch.cross(-normals, grasp_axe)
    # rots = torch.stack([grasp_axe, y, -normals], dim=2)
    if collision_check:
        volume = sample['sdf_volume']
        origin = sample['origin']
        resolution = sample['resolution']
        col_flag = collision_checking(volume, origin, resolution, pts, normals, rots, gripper_points)
        pts = pts[col_flag]
        normals = normals[col_flag]
        print(col_flag.shape)
        print(rots.shape)
        rots = rots[col_flag]
        print(rots.shape)
        scores = scores[col_flag]
    # transfer to numpy array
    pts = pts.detach().cpu().numpy()
    normals = normals.detach().cpu().numpy()
    rots = rots.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    r = R.from_matrix(rots)
    quats = r.as_quat()  # in x,y,z,w order
    poses = np.concatenate([pts, normals, quats], axis=1)  # [0:3], [3:6], [6:10]
    groups = group_and_select_poses(poses, scores, dist_threshold)
    return groups


def collision_checking(volume, origin, resolution,
                       pos, normal, rot, gpr_pts,
                       approaching_dist=0.025, th_col=1, th_sdf=-0.1):
    """
    Collision checking for all the grasp candidates represented by position, surface normal, and rotation matrix.
    In this function, we project gripper points into the volume by the gripper pose and examine sign of values to 
    check if current grasp candidate is collided with objects or not.
    We assume the origin of gripper frame is located at the center between two finger tips. Just like the panda gripper.
    :param volume: An B*C*H*W*D-d torch tensor representing the sdf volume.
    :param origin: B*3-d
    :param resolution: B-d
    :param pos: An Nx3-d torch tensor representing positions of all the grasp candidates.
    :param normal: An Nx3-d torch tensor representing inverse of approaching vectors of all the grasp candidates.
    :param rot: An Nx3x3-d torch tensor representing orientations of all the grasp candidates.
    :param gpr_pts: An Nx3-d torch tensor representing points of a gripper
    :param approaching_dist: Approaching distance
    :param th_col: The minimum number of collided voxels allowed.
    :param th_sdf: The threshold of minimum collision free sdf value.
    :return: An N-d boolean tensor specifying which candidates are collision-free
    """
    num_pose = pos.shape[0]
    tran = pos - normal * approaching_dist  # use -normal here to represent the approaching direction
    gpr_pts = torch.stack([gpr_pts]*num_pose, dim=0)  # num_pose * num_gpr_pts * 3
    gpr_pts = torch.matmul(gpr_pts, rot.permute(0, 2, 1)) + tran.view(num_pose, 1, 3)
    gpr_pts = torch.unsqueeze(gpr_pts.permute(2, 1, 0), dim=0)
    sdf_val = extract_sdf(gpr_pts, volume, origin, resolution)
    col_map = sdf_val <= th_sdf
    num_collided = torch.squeeze(torch.sum(torch.squeeze(col_map), dim=0))
    obj_flag = num_collided <= th_col
    th_bkg = torch.tensor(2, dtype=torch.float32).to(gpr_pts.device)
    bkg_flag = torch.logical_not(torch.logical_and(pos[:, 2] - origin[0, 2] < 0.025,
                                                   normal[:, 2] <= torch.cos(torch.deg2rad(th_bkg))))
    flags = obj_flag * bkg_flag
    num_valid = torch.sum(flags)
    if num_valid == 0:
        print('all candidates are collided with objects, choose one with minimal collision !!!')
        flags = num_collided == torch.min(num_collided)
    if flags.shape == torch.Size([]):
        flags = torch.unsqueeze(flags, dim=0)

    return flags


def group_and_select_poses(poses, scores, threshold=0.02):
    def create_a_group(pose, priority):
        d = {'center': pose[0:3],
             'num_pts': 1,
             'queue': PriorityQueue()}
        d['queue'].push(pose, priority)
        return d
    g = dict()
    num_group = 0
    num_pose = poses.shape[0]
    for i in range(num_pose):
        if not g:
            g[num_group] = create_a_group(poses[i], float(scores[i]))
            num_group += 1
            continue
        j = 0
        while j < num_group:
            if np.linalg.norm(g[j]['center']-poses[i][0:3]) < threshold:
                g[j]['center'] = (g[j]['num_pts'] * g[j]['center'] + poses[i][0:3]) / (g[j]['num_pts'] + 1)
                g[j]['num_pts'] += 1
                g[j]['queue'].push(poses[i], float(scores[i]))
                break
            j += 1
        if j == num_group:
            g[num_group] = create_a_group(poses[i], float(scores[i]))
            num_group += 1
    g = sorted(g.values(), key=lambda d: d['num_pts'], reverse=True)
    return g


class DiscretizedGripper(object):
    def __init__(self, cfg):
        self.num_sample = cfg['num_sample']
        self.hand_outer_diameter = cfg['hand_outer_diameter']
        self.hand_height = cfg['hand_height']
        self.hand_depth = cfg['hand_depth']
        self.finger_width = cfg['finger_width']
        self.inner_diameter = self.hand_outer_diameter - 2 * self.finger_width
        sample_ids = np.arange(self.num_sample)
        ids = np.stack(np.meshgrid(sample_ids, sample_ids, sample_ids, indexing='ij'), axis=3).reshape(-1, 3)  # N * 3
        ids = ids / self.num_sample - 0.5  # [-0.5, 0.5)
        body_shape = np.array([self.hand_outer_diameter,
                               self.hand_height,
                               self.finger_width], dtype=np.float32).reshape(1, 3)
        left_shape = right_shape = np.array([self.finger_width,
                                             self.hand_height,
                                             self.hand_depth], dtype=np.float32).reshape(1, 3)
        inner_shape = np.array([self.inner_diameter,
                                self.hand_height,
                                self.hand_depth], dtype=np.float32).reshape(1, 3)
        left_origin = np.array([-self.hand_outer_diameter / 2 + self.finger_width / 2,
                                0,
                                self.finger_width / 2 + self.hand_depth / 2], dtype=np.float32).reshape(1, 3)
        right_origin = np.array([self.hand_outer_diameter / 2 - self.finger_width / 2,
                                 0,
                                 self.finger_width / 2 + self.hand_depth / 2], dtype=np.float32).reshape(1, 3)
        inner_origin = np.array([0,
                                 0,
                                 self.finger_width / 2 + self.hand_depth / 2], dtype=np.float32).reshape(1, 3)
        body = ids * body_shape
        left = left_origin + ids * left_shape
        right = right_origin + ids * right_shape
        inner = inner_origin + ids * inner_shape
        self._inner = inner  # N * 3
        self._base = np.concatenate([body, left, right, inner], axis=0).astype(np.float32)  # 4N * 3
        self.num_pts = self._base.shape[0]
        # random perturbation
        self.rand_x = partial(np.random.uniform, cfg['x_min'], cfg['x_max'])
        self.rand_y = partial(np.random.uniform, cfg['y_min'], cfg['y_max'])
        self.rand_z = partial(np.random.uniform, cfg['z_min'], cfg['z_max'])
        self.rand_a = partial(np.random.uniform,
                              np.deg2rad(cfg['alpha_min']),
                              np.deg2rad(cfg['alpha_max']))
        self.rand_b = partial(np.random.uniform,
                              np.deg2rad(cfg['beta_min']),
                              np.deg2rad(cfg['beta_max']))
        self.rand_g = partial(np.random.uniform,
                              np.deg2rad(cfg['gamma_min']),
                              np.deg2rad(cfg['gamma_max']))

    def sample_perturbation(self, num_pose):
        delta_trans = np.stack([self.rand_x(num_pose),
                                self.rand_y(num_pose),
                                self.rand_z(num_pose)], axis=1).astype(np.float32)
        delta_euler = np.stack([self.rand_g(num_pose),
                                self.rand_b(num_pose),
                                self.rand_a(num_pose)], axis=1).astype(np.float32)
        delta_rot = euler2rot(delta_euler, seq='ZYX')
        return delta_trans, delta_rot

    def sample_perturbed_grippers(self, trans, rot, inner_only=False):
        """
        Sample points of grippers with perturbation given ground truth translations and rotation matrices
        :param trans: An N*3-d numpy array representing the positions of the grippers
        :param rot: An N*3*3-d numpy array representing the rotation matrices of grippers
        :param inner_only: Whether to sample points for inner region only
        :return: Two N*self.num_pts*3-d numpy arrays representing the positions of
        sample points of grippers without and with perturbation respectively,
        an N*3-d numpy array representing the perturbed translations,
        and an N*3*3-d numpy representing the perturbed rotation matrices.
        """
        pos = self.sample_grippers(trans, rot, inner_only)
        num_pose = trans.shape[0]
        delta_trans, delta_rot = self.sample_perturbation(num_pose)
        perturbed_pos = np.matmul(pos, delta_rot.transpose((0, 2, 1))) + delta_trans.reshape((num_pose, 1, 3))
        return pos, perturbed_pos, delta_trans, delta_rot

    def sample_grippers(self, trans, rot, inner_only=False):
        assert trans.shape[0] == rot.shape[0]
        num_pose = trans.shape[0]
        if inner_only:
            base_pos = np.stack([self._inner] * num_pose, axis=0)  # num_pose * self.num_pts * 3
        else:
            base_pos = np.stack([self._base] * num_pose, axis=0)  # num_pose * self.num_pts * 3
        pos = np.matmul(base_pos, rot.transpose((0, 2, 1))) + trans.reshape(num_pose, 1, 3)
        return pos.astype(np.float32)
