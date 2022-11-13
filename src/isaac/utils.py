import os
import json
from sim.utils import basic_rot_mat
from isaacgym import gymapi
from sim.camera import *
import pymeshlab as ml
import torch
from scipy.spatial.transform import Rotation as R
from vgn.utils.transform import Rotation
import numpy as np


class Camera_ig(object):
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


def vpn_predict(tsdf, vpn, dg, rn, gpr_pts, device, use_rn=False):
    from vpn.utils import rank_and_group_poses
    from src.sim.utils import basic_rot_mat

    v, _, n, _ = tsdf.compute_mesh(step_size=2)
    v, n = torch.from_numpy(v).to(device).to(torch.float32), torch.from_numpy(n).to(device).to(torch.float32)
    ids = tsdf.get_ids(v)
    sample = dict()
    sample['sdf_volume'] = tsdf.gaussian_blur(tsdf.post_processed_volume).unsqueeze(dim=0).unsqueeze(dim=0)
    sample['pts'] = v.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    sample['normals'] = n.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    sample['ids'] = ids.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    sample['occupancy_volume'] = torch.zeros_like(sample['sdf_volume'], device=device)
    sample['occupancy_volume'][sample['sdf_volume'] <= 0] = 1
    sample['origin'] = tsdf.origin
    sample['resolution'] = torch.tensor([[tsdf.resolution]], dtype=torch.float32, device=device)
    out = torch.sigmoid(vpn.forward(sample))
    groups = rank_and_group_poses(sample, out, device, gpr_pts, collision_check=True)
    if groups is None:
        return None, None, None

    score, pose = groups[0]['queue'].get()
    pos, rot0 = pose[0:3], R.from_quat(pose[6:10]).as_matrix()
    rot = rot0 @ basic_rot_mat(np.pi / 2, axis='z').astype(np.float32)
    quat = R.from_matrix(rot).as_quat()
    gripper_pos = pos - 0.08 * rot[:, 2]
    if use_rn:
        for _ in range(5):
            trans = (pose[0:3] + pose[3:6] * 0.01).astype(np.float32)
            rot = R.from_quat(pose[6:10]).as_matrix().astype(np.float32)
            pos = dg.sample_grippers(trans.reshape(1, 3), rot.reshape(1, 3, 3), inner_only=False)
            pos = np.expand_dims(pos.transpose((2, 1, 0)), axis=0)  # 1 * 3 * 2048 * 1
            sample['perturbed_pos'] = torch.from_numpy(pos).to(device)
            delta_rot = rn(sample)
            delta_rot = torch.squeeze(delta_rot).detach().cpu().numpy()
            rot_recover = rot @ delta_rot
            approach_recover = rot_recover[:, 2]
            quat_recover = R.from_matrix(rot_recover).as_quat()
            pos_recover = pose[0:3]
            pose = np.concatenate([pos_recover, -approach_recover, quat_recover])
        pos, rot0 = pose[0:3], R.from_quat(pose[6:10]).as_matrix()
        rot = rot0 @ basic_rot_mat(np.pi / 2, axis='z').astype(np.float32)
        quat = R.from_matrix(rot).as_quat()
        gripper_pos = pos - 0.08 * rot[:, 2]
    return gripper_pos, rot, quat


def cpn_predict(tsdf, cpn, pg, cfg):
    from cpn.utils import sample_contact_points, select_gripper_pose, clustering
    cp1s, cp2s = sample_contact_points(tsdf)
    if len(cp1s) == 0:
        return None, None, None, None, None
    ids_cp1, ids_cp2 = tsdf.get_ids(cp1s), tsdf.get_ids(cp2s)

    sample = dict()
    sample['sdf_volume'] = tsdf.gaussian_smooth(tsdf.post_processed_vol).unsqueeze(dim=0).unsqueeze(dim=0)
    sample['ids_cp1'] = ids_cp1.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    sample['ids_cp2'] = ids_cp2.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
    out = torch.squeeze(cpn.forward(sample))

    gripper_poses, rots, widths, cp1s_, cp2s_ = select_gripper_pose(tsdf, pg.vertex_sets,
                                                                    out, cp1s, cp2s, cfg['gripper']['depth'],
                                                                    check_tray=True,
                                                                    th_s=0.997,
                                                                    th_c=0.999)

    if gripper_poses is None:
        return None, None, None, None, None
    pos, rot, width, cp1, cp2 = clustering(gripper_poses, rots, widths, cp1s_, cp2s_)
    quat = R.from_matrix(rot).as_quat()

    return pos, rot, quat, cp1, cp2


def acquire_tsdf(low_res_tsdf, high_res_tsdf):
    tsdf = low_res_tsdf
    pc = high_res_tsdf.get_cloud()
    return tsdf, pc


def select_grasp(grasps, scores):
    # select the highest grasp
    heights = np.empty(len(grasps))
    for i, grasp in enumerate(grasps):
        heights[i] = grasp.pose.translation[2]
    idx = np.argmax(heights)
    grasp, score = grasps[idx], scores[idx]

    # make sure camera is pointing forward
    rot = grasp.pose.rotation
    axis = rot.as_matrix()[:, 0]
    if axis[0] < 0:
        grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)
    return grasp, score


def get_grasp_pose(obj_name,
                   data_path=None
                   ):
    if data_path is None:
        data_path = '/home/sujc/code/vcpd-master/dataset/grasp_info/train_grasp_info'
    json_file = os.path.join(data_path, '{}_info.json'.format(obj_name))
    if os.path.isfile(json_file):
        with open(json_file, 'r', encoding='utf8') as f:
            json_data = json.load(f)
        if json_data['num_col-free_poses'] == 0:
            return None, None, None, None
    positions = []
    quats = []
    rots = []
    gripper_positions = []
    centers = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'centers')))
    widths = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'widths')))
    collisions = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'collisions')))
    quaternions = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'quaternions')))
    num_angle = collisions.shape[1]
    if quaternions.shape[1] == 4:
        angles = np.arange(num_angle) / num_angle * 2 * np.pi
        basic_rot_mats = np.expand_dims(basic_rot_mat(angles, 'y'), axis=0)  # 1*num_angle*3*3
        bases0 = np.expand_dims(Rotation.from_quat(quaternions).as_matrix(), axis=1)  # 1*n*3*3
        rot0 = np.matmul(bases0, basic_rot_mats) # n*64*3*3

    scores = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'antipodal_mean')))
    for i in range(centers.shape[0]):
        grasp_ids = np.where(collisions[i] == 0)[0]
        if grasp_ids.shape[0] == 0:
            continue
        center = centers[i]
        quaternion = quaternions[i]
        # score = scores[i]
        # if score < 0.99:
        #     continue
        # if len(positions)>500:
        #     break
        for j in range(grasp_ids.shape[0]):
            grasp_id = grasp_ids[j]

            if quaternions.shape[1] == num_angle:
                quat = quaternion[grasp_id]
                rot = R.from_quat(quat).as_matrix()
            else:
                rot = rot0[i][grasp_id]
                quat = R.from_matrix(rot).as_quat()
            pos = center.copy()
            gripper_pos = center.copy() - rot[:, 2] * 0.10327

            positions.append(pos)
            quats.append(quat)
            rots.append(rot)
            gripper_positions.append(gripper_pos)
    positions = np.array(positions)
    quats = np.array(quats)
    rots = np.array(rots)
    gripper_positions = np.array(gripper_positions)
    return positions, quats, rots, gripper_positions

def asset_preload(dir, num, filename=None, info_path=None):
    asset_paths = []
    asset_names = []
    positions = []
    quats = []
    rots = []
    gripper_positions = []
    count = 0
    n_max = 0

    for files in os.listdir(dir):
        if filename is not None:
            file = os.path.join(dir, filename)
        else:
            file = os.path.join(dir, files)

        if os.path.isfile(file) and file.find('.urdf') > 0:
            i = file.rfind('/')
            asset_paths.append(file[i+1:])

            name = file[i+1: -5]
            asset_names.append(name)

            position, quat, rot, gripper_pos = get_grasp_pose(name,info_path)
            if position is None:
                return None, None, None, None, None, None
            if len(position) > n_max:
                n_max = len(position)
            positions.append(position)
            quats.append(quat)
            rots.append(rot)
            gripper_positions.append(gripper_pos)
            count += 1
            if count == num:
                break
        if filename is not None:
            break
    for i in range(count):
        # n = n_max - len(positions[i])
        # for j in range(n):
        #     positions[i].append(np.array([100] * 3))
        #     gripper_positions[i].append(np.array([100] * 3))
        #     quats[i].append(np.array([100] * 4))
        #     rots[i].append( 100 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) )
        positions[i] = np.array(positions[i])
        quats[i] = np.array(quats[i])
        rots[i] = np.array(rots[i])
        gripper_positions[i] = np.array(gripper_positions[i])
    try:
        positions = np.squeeze(np.array(positions).transpose((1,0,2)))
        quats = np.squeeze(np.array(quats).transpose((1,0,2)))
        rots = np.squeeze(np.array(rots).transpose((1,0,2,3)))
        gripper_positions = np.squeeze(np.array(gripper_positions).transpose((1,0,2)))
        if len(positions.shape) == 1:
            positions = np.expand_dims(positions,axis=0)
            quats = np.expand_dims(quats, axis=0)
            rots = np.expand_dims(rots, axis=0)
            gripper_positions = np.expand_dims(gripper_positions, axis=0)
    except:
        print('error')
        print(filename)
    return asset_paths, asset_names, positions, quats, rots, gripper_positions

def get_grasp_pose_primitive(obj_name,
                   data_path=None
                   ):
    if data_path is None:
        data_path = '/home/sujc/code/vcpd-master/dataset/grasp_info/train_grasp_info'
    json_file = os.path.join(data_path, '{}_info.json'.format(obj_name))
    if os.path.isfile(json_file):
        with open(json_file, 'r', encoding='utf8') as f:
            json_data = json.load(f)
        if json_data['num_col-free_poses'] == 0:
            return None, None, None, None

    centers = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'centers')))
    collisions = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'collisions')))
    quaternions = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'quaternions')))
    widths = np.load(os.path.join(data_path, '{}_{}.npy'.format(obj_name, 'widths')))

    num_angle = collisions.shape[1]
    centers = np.concatenate([np.expand_dims(centers, axis=1)] * 64, axis=1)
    widths = np.concatenate([np.expand_dims(widths, axis=1)] * 64, axis=1)
    if quaternions.shape[1] == 4:
        angles = np.arange(num_angle) / num_angle * 2 * np.pi
        basic_rot_mats = np.expand_dims(basic_rot_mat(angles, 'y'), axis=0)  # 1*num_angle*3*3
        bases0 = np.expand_dims(Rotation.from_quat(quaternions).as_matrix(), axis=1)  # 1*n*3*3
        gripper_rot = np.matmul(bases0, basic_rot_mats) # n*64*3*3
    else:
        gripper_rot = Rotation.from_quat(quaternions).as_matrix()
    gripper_pos = centers - gripper_rot[..., 2] * 0.10327

    flag = collisions == 0

    return centers[flag], gripper_rot[flag], gripper_pos[flag], widths[flag], collisions


def asset_preload_primitive(dir, obj_name, info_path):
    count = 0

    for file in os.listdir(os.path.join(dir, obj_name)):
        if 'urdf' in file:
            asset_path = file

    position, rot, gripper_pos, width, collision = get_grasp_pose_primitive(obj_name,info_path)
    if position is None:
        return None, None, None, None, None, None
    count += 1

    return asset_path, position, rot, gripper_pos, width, collision

def load_panda(gym, sim):
    asset_root = "/home/sujc/code/isaacgym/assets"
    panda_asset_file = "urdf/franka_description/robots/panda.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    # asset_options.override_com = True
    # asset_options.override_inertia = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    panda_asset = gym.load_asset(sim, asset_root, panda_asset_file, asset_options)
    obj_prop = gymapi.RigidShapeProperties()
    obj_prop.friction = 100.0
    obj_prop.restitution = 0.9
    obj_prop.rolling_friction = 100.0
    # gym.set_asset_rigid_shape_properties(panda_asset, [obj_prop])
    # configure panda dofs
    panda_dof_props = gym.get_asset_dof_properties(panda_asset)
    panda_lower_limits = panda_dof_props["lower"]
    panda_upper_limits = panda_dof_props["upper"]
    panda_ranges = panda_upper_limits - panda_lower_limits

    # grippers
    panda_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    panda_dof_props["stiffness"].fill(800.0)
    panda_dof_props["damping"].fill(40.0)

    # default dof states and position targets
    panda_num_dofs = gym.get_asset_dof_count(panda_asset)
    default_dof_pos = np.zeros(panda_num_dofs, dtype=np.float32)
    # grippers open
    default_dof_pos = panda_upper_limits

    default_dof_state = np.zeros(panda_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    return default_dof_pos, default_dof_state, panda_asset, panda_dof_props



def get_obj_file(path):
    '''
    example:
    file -> obj_name: 'A00_0.urdf' -> A00_0; 'C2.urdf' -> C2
    '''
    obj_files = []
    file= open(path,"r")
    for line in file.readlines():
        line = line.strip('\n')
        obj_files.append(line)
    file.close()
    return obj_files

def get_tray(shift=gymapi.Vec3(), height=0.384, width=0.384, depth=0.1, theta=45, thickness=0.01):
    d2r = np.deg2rad
    h, w, d, t = height / 2, width / 2, depth / 2, thickness / 2
    assert 0 < theta <= 90
    side_height = depth / np.sin(d2r(theta))
    s_h = side_height / 2
    e = 0 if theta == 90 else depth / np.tan(d2r(theta))  # half extend length for each side of groove
    # compute the normal and tangent components of half thickness
    n_com, t_com = t * np.cos(d2r(theta)), t * np.sin(d2r(theta))
    size_list = 2*np.array([[h, w, t], [h + e, s_h, t], [h + e, s_h, t], [s_h, w + e, t], [s_h, w + e, t]])
    size_list = size_list.tolist()
    pos_list = [[0, 0, t],
                 [0, -w - e / 2 - t_com, d + thickness - n_com],
                 [0, w + e / 2 + t_com, d + thickness - n_com],
                 [-h - e / 2 - t_com, 0, d + thickness - n_com],
                 [h + e / 2 + t_com, 0, d + thickness - n_com]]
    quat_list=[[0, 0, 0, 1],
               R.from_euler('xyz',[d2r(-theta), d2r(0), d2r(0)]).as_quat(),
               R.from_euler('xyz',[d2r(theta), d2r(0), d2r(0)]).as_quat(),
               R.from_euler('xyz',[d2r(0), d2r(theta), d2r(0)]).as_quat(),
               R.from_euler('xyz',[d2r(0), -d2r(theta), d2r(0)]).as_quat()]
    pose_list = []

    for i in range(5):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(pos_list[i][0], pos_list[i][1], pos_list[i][2]) + shift
        pose.r = gymapi.Quat(quat_list[i][0], quat_list[i][1], quat_list[i][2], quat_list[i][3])
        pose_list.append(pose)
    return size_list, pose_list

def add_noise(depth,
              intrinsic,
              lateral=True,
              axial=True,
              missing_value=True,
              default_angle=85.0):
    """
    Add noise according to kinect noise model.
    Please refer to the paper "Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking".
    """
    h, w = depth.shape
    point_cloud = compute_point_cloud(depth, intrinsic)
    surface_normal = compute_surface_normal_central_difference(point_cloud)
    # surface_normal = self.compute_surface_normal_least_square(point_cloud)
    cos = np.squeeze(np.dot(surface_normal, np.array([[0.0, 0.0, 1.0]], dtype=surface_normal.dtype).T))
    angles = np.arccos(cos)
    # adjust angles that don't satisfy the domain of noise model ([0, pi/2) for kinect noise model).
    cos[angles >= np.pi / 2] = np.cos(np.deg2rad(default_angle))
    angles[angles >= np.pi / 2] = np.deg2rad(default_angle)
    # add lateral noise
    if lateral:
        sigma_lateral = 0.8 + 0.035 * angles / (np.pi / 2 - angles)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        # add noise offset to x axis
        new_x = x + np.round(np.random.normal(scale=sigma_lateral)).astype(np.int)
        # remove points that are out of range
        invalid_ids = np.logical_or(new_x < 0, new_x >= w)
        new_x[invalid_ids] = x[invalid_ids]
        # add noise offset to y axis
        new_y = y + np.round(np.random.normal(scale=sigma_lateral)).astype(np.int)
        # remove points that are out of range
        invalid_ids = np.logical_or(new_y < 0, new_y >= h)
        new_y[invalid_ids] = y[invalid_ids]
        depth = depth[new_y, new_x]
    # add axial noise
    if axial:
        # axial noise
        sigma_axial = 0.0012 + 0.0019 * (depth - 0.4) ** 2
        depth = np.random.normal(depth, sigma_axial)
    # remove some value according to the angle
    # the larger the angle, the higher probability the depth value is set to zero
    if missing_value:
        missing_mask = np.random.uniform(size=cos.shape) > cos
        depth[missing_mask] = 0.0
    return depth

def compute_point_cloud(depth, intrinsic):
    """
    Compute point cloud by depth image and camera intrinsic matrix.
    :param depth: A float numpy array representing the depth image.
    :param intrinsic: A 3x3 numpy array representing the camera intrinsic matrix
    :return: Point cloud in camera space.
    """
    h, w = depth.shape
    h_map, w_map = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    image_coordinates = np.stack([w_map, h_map, np.ones_like(h_map, dtype=np.float32)], axis=2).astype(np.float32)
    inv_intrinsic = np.linalg.inv(intrinsic)
    camera_coordinates = np.expand_dims(depth, axis=2) * np.dot(image_coordinates, inv_intrinsic.T)
    return camera_coordinates

def compute_surface_normal_central_difference(point_cloud):
    """
    Compute surface normal from point cloud.
    Notice: it only applies to point cloud map represented in camera space.
    The x axis directs in width direction, and y axis is in height direction.
    :param point_cloud: An HxWx3-d numpy array representing the point cloud map.The point cloud map
                        is restricted to the map in camera space without any other transformations.
    :return: An HxWx3-d numpy array representing the corresponding normal map.
    """
    h, w, _ = point_cloud.shape
    gradient_y, gradient_x, _ = np.gradient(point_cloud)
    normal = np.cross(gradient_x, gradient_y, axis=2)
    normal[normal == np.nan] = 0
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    flag = norm[..., 0] != 0
    normal[flag] = normal[flag] / norm[flag]
    return normal

class PandaGripper(object):
    def __init__(self, asset_path):
        self.components = ['hand', 'left_finger', 'right_finger']
        self.vertex_sets = dict()
        ms = ml.MeshSet()
        for component in self.components:
            col2_path = os.path.join(asset_path, component+'_col2.obj')
            ms.load_new_mesh(col2_path)
            self.vertex_sets[component] = ms.current_mesh().vertex_matrix()

class Robotiq(object):
    def __init__(self, asset_path):
        self.components = ['rq85']
        self.vertex_sets = dict()
        ms = ml.MeshSet()
        for component in self.components:
            col_path = os.path.join(asset_path, component+'.obj')
            ms.load_new_mesh(col_path)
            self.vertex_sets[component] = ms.current_mesh().vertex_matrix()