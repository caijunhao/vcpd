from sim.utils import *
from sim.objects import PandaGripper, RigidObject
from sim.camera import Camera
from sim.tray import Tray
from sdf import TSDF, PSDF, GradientSDF
from scipy.spatial.transform import Rotation
from skimage import io
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import numpy as np
import torch
import trimesh
import argparse
import shutil
import json
import time
import os

np.random.seed(7)
torch.set_default_dtype(torch.float32)
tf32 = torch.float32
nf32 = np.float32


def check_valid_pts(pts, bounds):
    """
    check if the given points are inside the boundaries or not.
    :param pts: an Nx3-d numpy array representing the 3-d points.
    :param bounds: a 2x3-d numpy array representing the boundaries of the volume.
    :return: an N-d numpy array representing if the points are inside the given boundaries or not.
    """
    flag_x = np.logical_and(pts[:, 0] >= bounds[0, 0], pts[:, 0] <= bounds[1, 0])
    flag_y = np.logical_and(pts[:, 1] >= bounds[0, 1], pts[:, 1] <= bounds[1, 1])
    flag_z = np.logical_and(pts[:, 2] >= bounds[0, 2], pts[:, 2] <= bounds[1, 2])
    flag = flag_x * flag_y * flag_z
    return flag


def get_neg_from_pos(cp1, cp2):
    dtype = cp1.dtype
    num_cp = cp1.shape[0]
    if num_cp <= 1:
        return np.zeros((0, 3), dtype=dtype), np.zeros((0, 3), dtype=dtype)
    y = cp1 - cp2
    norm_y = np.linalg.norm(y, axis=1)
    # y = y / norm_y
    ids = np.arange(num_cp)
    np.random.shuffle(ids)
    ncp1 = cp1
    ncp2 = cp2[ids]
    ny = ncp1 - ncp2
    norm_ny = np.linalg.norm(ny, axis=1)
    # ny = ny / norm_ny
    norm = norm_y * norm_ny
    norm_flag = norm > 0
    cos_wo_norm = np.sum(y * ny, axis=1)  # cos_w_norm = np.sum((y / norm) * (ny / norm_ny), axis=1)
    # random samples whose angles between the random and original grasp direction are larger than
    # 17 degree will be considered as valid negative samples.
    cos_flag = np.cos(np.deg2rad(17)) * norm > cos_wo_norm
    flag = norm_flag * cos_flag
    ncp1, ncp2 = ncp1[flag], ncp2[flag]
    return ncp1, ncp2


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(args.config, 'r') as config_file:
        cfg = json.load(config_file)
    # PyBullet initialization
    mode = p.GUI if args.gui else p.DIRECT
    p.connect(mode)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF('plane.urdf')
    # scene initialization
    tray = Tray(**cfg['tray'])
    tray.set_pose([-1, -1, -1])
    cfg['tray']['height'], cfg['tray']['width'], cfg['tray']['theta'] = 0.324, 0.324, 80
    init_tray = Tray(**cfg['tray'])
    cam = Camera(**cfg['camera'])
    pg = PandaGripper('assets')
    pg.set_pose([-2, -2, -2], [0, 0, 0, 1])
    exemption = tray.get_tray_ids()
    mesh_list = os.listdir(args.mesh)
    angles = np.arange(cfg['num_angle']) / cfg['num_angle'] * 2 * np.pi
    basic_rot_mats = np.expand_dims(basic_rot_mat(angles, 'y'), axis=0)  # 1*num_angle*3*3
    # sdf initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxel_length = cfg['sdf']['voxel_length']
    vol_bnd = np.array([cfg['sdf']['x_min'], cfg['sdf']['y_min'], cfg['sdf']['z_min'],
                        cfg['sdf']['x_max'], cfg['sdf']['y_max'], cfg['sdf']['z_max']]).reshape(2, 3)
    origin = vol_bnd[0]
    resolution = np.ceil((vol_bnd[1] - vol_bnd[0] - voxel_length / 7) / voxel_length).astype(np.int)
    sdf = TSDF(origin, resolution, voxel_length, fuse_color=True, device=device)
    sdf2 = TSDF(origin, resolution, voxel_length, fuse_color=False, device=device)
    i = 0
    while i < cfg['scene']['trial']:
        info_dict = dict()
        dynamic_list = list()
        static_list = list()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for j in range(cfg['scene']['obj']):
            mesh_name = np.random.choice(mesh_list)
            if mesh_name not in info_dict:
                with open(os.path.join(args.info, mesh_name + '_info.json'), 'r') as f:
                    keys = json.load(f)['keys']
                info_dict[mesh_name] = {k: np.load(os.path.join(args.info, mesh_name + '_' + k + '.npy')) for k in keys}
            print('loading {} into pybullet...'.format(mesh_name))
            vis_params, col_params, body_params = get_multi_body_template()
            vis_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_vis.obj')
            col_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_col.obj')
            pos, quat = np.array([1 - 0.5 * j, 1, 0.1]), np.array([0, 0, 0, 1])
            body_params['basePosition'], body_params['baseOrientation'] = pos, quat
            body_params['baseMass'] = 1.0
            dynamic_list.append(RigidObject(mesh_name,
                                            vis_params=vis_params,
                                            col_params=col_params,
                                            body_params=body_params))
            # dynamic_list.append(RigidObject(mesh_name, urdf_path=os.path.join(args.mesh, mesh_name, mesh_name+'.urdf')))
            body_params['baseMass'] = 0
            body_params['basePosition'] = np.array([1 - 0.5 * j, 2, 0.1])
            static_list.append(RigidObject(mesh_name,
                                           vis_params=vis_params,
                                           col_params=col_params,
                                           body_params=body_params))
            static_list[-1].change_color()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.resetDebugVisualizerCamera(cameraDistance=0.57, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0, 0, 0])
        place_order = np.arange(cfg['scene']['obj'])
        np.random.shuffle(place_order)
        for j in place_order:
            pos, quat = sample_a_pose(**cfg['pose_range'])
            dynamic_list[j].set_pose(pos, quat)
            step_simulation(30)
        stable = False
        curr_wait = 0
        while not stable and curr_wait < cfg['scene']['wait']:
            stable = dynamic_list[0].is_stable() & dynamic_list[-1].is_stable()
            curr_wait += 1
        init_tray.set_pose([-2, 0, 0])
        tray.set_pose([0, 0, 0])
        stable = False
        curr_wait = 0
        while not stable and curr_wait < 3:
            stable = dynamic_list[0].is_stable() & dynamic_list[-1].is_stable()
            curr_wait += 1
        print('scene stable, start generating data.')
        poses = np.zeros((cfg['scene']['obj'], 7))
        contact_pts1, contact_pts2, neg_pts1, neg_pts2 = list(), list(), list(), list()
        for j in place_order:
            pos, quat = dynamic_list[j].get_pose()
            dynamic_list[j].set_pose(np.array([1 - 0.5 * j, 1, 0.1]), np.array([0, 0, 0, 1]))
            static_list[j].set_pose(pos, quat)
            poses[j, :3], poses[j, 3:] = pos, quat
        # pose filtering and collision checking
        b = time.time()
        for o in static_list:
            pose = o.transform
            if 'collision_refine' in info_dict[o.obj_name]:
                col_flag = np.logical_not(info_dict[o.obj_name]['collision_refine'])
            else:
                col_flag = np.logical_not(info_dict[o.obj_name]['collisions'])
            candidate_flag = np.logical_not(col_flag[col_flag])
            bases0 = np.expand_dims(Rotation.from_quat(info_dict[o.obj_name]['quaternions']).as_matrix(), axis=1)  # n*1*3*3
            rots0 = np.matmul(bases0, basic_rot_mats)[col_flag]
            centers0 = np.stack([info_dict[o.obj_name]['centers']] * cfg['num_angle'], axis=1)[col_flag]
            intersects0 = np.stack([info_dict[o.obj_name]['intersects']] * cfg['num_angle'], axis=1)[col_flag]
            w0 = np.stack([info_dict[o.obj_name]['widths']] * cfg['num_angle'], axis=1)[col_flag]
            rots = np.matmul(pose[0:3, 0:3].reshape(1, 3, 3), rots0)
            quats = Rotation.from_matrix(rots).as_quat()
            centers = centers0 @ pose[0:3, 0:3].T + pose[0:3, 3].reshape(1, 3)
            intersects = intersects0 @ pose[0:3, 0:3].T + pose[0:3, 3].reshape(1, 3)
            contacts = centers + (centers - intersects)
            indices = np.stack([np.arange(col_flag.shape[0])] * cfg['num_angle'], axis=1)[col_flag]
            z_axes = rots[..., :, 2]
            z_flag = z_axes[..., 2] < 0
            curr_idx = -1
            for k in np.arange(z_flag.shape[0])[z_flag]:
                if indices[k] == curr_idx:
                    continue
                pos = centers[k] - z_axes[k] * cfg['gripper']['depth']
                pg.set_pose(pos, quats[k])
                pg.set_gripper_width(w0[k] + 0.02)
                candidate_flag[k] = not pg.is_collided(exemption)
                curr_idx = indices[k] if candidate_flag[k] else curr_idx
            contacts1 = contacts[candidate_flag]
            contacts2 = intersects[candidate_flag]
            contact_pts1.append(contacts1)
            contact_pts2.append(contacts2)
            # generate negative contact point pairs from positive ones
            neg_contacts1, neg_contacts2 = get_neg_from_pos(contacts1, contacts2)
            neg_pts1.append(neg_contacts1)
            neg_pts2.append(neg_contacts2)
            # retrieve negative contact points
            if 'collision_refine' in info_dict[o.obj_name]:
                neg_flag = np.sum(info_dict[o.obj_name]['collision_refine'], axis=1) == cfg['num_angle']
            else:
                neg_flag = np.sum(info_dict[o.obj_name]['collisions'], axis=1) == cfg['num_angle']
            neg_intersects0 = info_dict[o.obj_name]['intersects'][neg_flag]
            neg_centers0 = info_dict[o.obj_name]['centers'][neg_flag]
            neg_intersects = neg_intersects0 @ pose[0:3, 0:3].T + pose[0:3, 3].reshape(1, 3)
            neg_centers = neg_centers0 @ pose[0:3, 0:3].T + pose[0:3, 3].reshape(1, 3)
            neg_directions = neg_centers - neg_intersects
            neg_contacts = neg_centers + neg_directions
            # remove zero norm and normalize direction vector
            neg_norms = np.linalg.norm(neg_directions, axis=1, keepdims=True)
            neg_directions[:, 2][neg_norms[:, 0] == 0] = 1
            neg_norms[:, 0][neg_norms[:, 0] == 0] = 1
            neg_directions = neg_directions / np.linalg.norm(neg_directions, axis=1, keepdims=True)
            direction_flag = np.abs(neg_directions[:, 2]) <= np.cos(cfg['scene']['normal_threshold'])
            neg_contacts1 = neg_contacts[direction_flag]
            neg_contacts2 = neg_intersects[direction_flag]
            neg_pts1.append(neg_contacts1)
            neg_pts2.append(neg_contacts2)
        contact_pts1 = np.concatenate(contact_pts1, axis=0).astype(np.float32)
        contact_pts2 = np.concatenate(contact_pts2, axis=0).astype(np.float32)
        flag = check_valid_pts(contact_pts1, vol_bnd) * check_valid_pts(contact_pts2, vol_bnd)
        contact_pts1, contact_pts2 = contact_pts1[flag], contact_pts2[flag]
        neg_pts1 = np.concatenate(neg_pts1, axis=0).astype(np.float32)
        neg_pts2 = np.concatenate(neg_pts2, axis=0).astype(np.float32)
        flag = check_valid_pts(neg_pts1, vol_bnd) * check_valid_pts(neg_pts2, vol_bnd)
        neg_pts1, neg_pts2 = neg_pts1[flag], neg_pts2[flag]
        pg.set_pose([-2, -2, -2], [0, 0, 0, 1])
        e = time.time()
        print('elapse time on collision checking: {}s'.format(e - b))
        cam.set_pose(cfg['camera']['eye_position'], cfg['camera']['target_position'], cfg['camera']['up_vector'])
        path = os.path.join(args.output, '{:06d}'.format(i))
        os.makedirs(path, exist_ok=True)
        print('render scene images...')
        rgb_list, depth_list, pose_list, intr_list = list(), list(), list(), list()
        b = time.time()
        for j in range(cfg['scene']['frame']):
            rgb, depth, mask = cam.get_camera_image()
            rgb_list.append(rgb), depth_list.append(depth), pose_list.append(cam.pose), intr_list.append(cam.intrinsic)
            # noise_depth = camera.add_noise(depth)
            if args.scene:
                io.imsave(os.path.join(path, '{:04d}_rgb.png'.format(j)), rgb, check_contrast=False)
                io.imsave(os.path.join(path, '{:04d}_encoded_depth.png'.format(j)), cam.encode_depth(depth),
                          check_contrast=False)
                plt.imsave(os.path.join(path, '{:04d}_depth.png'.format(j)), depth)
                io.imsave(os.path.join(path, '{:04d}_mask.png'.format(j)), mask, check_contrast=False)
                np.save(os.path.join(path, '{:04d}_pos&quat.npy'.format(j)), np.concatenate(cam.get_pose()))
                np.save(os.path.join(path, '{:04d}_intrinsic.npy'.format(j)), cam.intrinsic)
            if cfg['scene']['pose_sampling'] == 'sphere':
                radius = np.random.uniform(cfg['camera']['z_min'], cfg['camera']['z_max'])
                cam.sample_a_pose_from_a_sphere(np.array(cfg['camera']['target_position']), radius)
            elif cfg['scene']['pose_sampling'] == 'cube':
                cam.sample_a_position(cfg['camera']['x_min'], cfg['camera']['x_max'],
                                      cfg['camera']['y_min'], cfg['camera']['y_max'],
                                      cfg['camera']['z_min'], cfg['camera']['z_max'],
                                      cfg['camera']['up_vector'])
            else:
                raise ValueError('only sphere or cube sampling methods are supported')
        e = time.time()
        print('elapse time on scene rendering: {}s'.format(e - b))
        # sdf generation
        sdf_path = os.path.join(args.output, '{:06d}'.format(i))
        discard = False
        if args.sdf:
            for ri, di, pi, ii, idx in zip(rgb_list, depth_list, pose_list, intr_list, range(len(intr_list))):
                os.makedirs(sdf_path) if not os.path.exists(sdf_path) else None
                ri, ndi = ri[..., 0:3], cam.add_noise(di)
                sdf.integrate(ndi, ii, pi, rgb=ri)
                sdf2.integrate(di, ii, pi, rgb=ri)
                if idx in cfg['sdf']['save_volume']:
                    sdf_cp1 = sdf2.interpolation(contact_pts1, smooth=False).cpu().numpy()
                    sdf_cp2 = sdf2.interpolation(contact_pts2, smooth=False).cpu().numpy()
                    th = 0.2
                    sdf_cp_flag = np.logical_and(np.abs(sdf_cp1) <= th, np.abs(sdf_cp2) <= th)
                    val_pts1, val_pts2 = contact_pts1[sdf_cp_flag], contact_pts2[sdf_cp_flag]
                    sdf_ncp1 = sdf2.interpolation(neg_pts1, smooth=False).cpu().numpy()
                    sdf_ncp2 = sdf2.interpolation(neg_pts2, smooth=False).cpu().numpy()
                    sdf_ncp_flag = np.logical_and(np.abs(sdf_ncp1) <= th, np.abs(sdf_ncp2) <= th)
                    val_n_pts1, val_n_pts2 = neg_pts1[sdf_ncp_flag], neg_pts2[sdf_ncp_flag]
                    num_cp = val_pts1.shape[0]
                    num_ncp = val_n_pts1.shape[0]
                    print('num_cp: {} | num_ncp: {}'.format(num_cp, num_ncp))
                    if num_cp < 10 or num_ncp < 10:
                        print('# of valid points are less than given threshold, discard current scene')
                        discard = True
                        break
                    else:
                        if cfg['sdf']['gaussian_blur']:
                            sdf_vol = sdf.gaussian_smooth(sdf.post_processed_vol)
                        else:
                            sdf_vol = sdf.post_processed_vol
                        sdf_vol_cpu = sdf_vol.cpu().numpy()
                        np.save(os.path.join(sdf_path, '{:04d}_pos_contact1.npy'.format(idx)),
                                sdf.get_ids(val_pts1).cpu().numpy())
                        np.save(os.path.join(sdf_path, '{:04d}_pos_contact2.npy'.format(idx)),
                                sdf.get_ids(val_pts2).cpu().numpy())
                        np.save(os.path.join(sdf_path, '{:04d}_neg_contact1.npy'.format(idx)),
                                sdf.get_ids(val_n_pts1).cpu().numpy())
                        np.save(os.path.join(sdf_path, '{:04d}_neg_contact2.npy'.format(idx)),
                                sdf.get_ids(val_n_pts2).cpu().numpy())
                        np.save(os.path.join(sdf_path, '{:04d}_sdf_volume.npy'.format(idx)),
                                sdf_vol_cpu)
        obj_info = [(o.obj_name, o.get_pose()) for o in static_list]
        np.save(os.path.join(sdf_path, '{:06d}_obj_info.npy'.format(i)), np.asanyarray(obj_info, dtype=object))
        sdf_info = {'voxel_length': cfg['sdf']['voxel_length'],
                    'origin': vol_bnd[0].tolist()}
        with open(os.path.join(sdf_path, '{:06d}_sdf_info.json'.format(i)), 'w') as fo:
            json.dump(sdf_info, fo, indent=4)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        [p.removeBody(o.obj_id) for o in dynamic_list]
        [p.removeBody(o.obj_id) for o in static_list]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        v, f, n, c = sdf.marching_cubes(step_size=3)
        m = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n, vertex_colors=c)
        m.export(os.path.join(sdf_path, '{:06d}_mesh.obj'.format(i)))
        # sdf.write_mesh(os.path.join(sdf_path, '{:06d}_mesh.ply'.format(i)), v, f, n, c)
        sdf.reset()
        sdf2.reset()
        if discard:
            continue
        i += 1
        if np.random.uniform() < 0.1:
            cam.randomize_fov()
            print('randomize fov: {}'.format(cam.curr_fov))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stacked scene construction.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to the config file.')
    parser.add_argument('--mesh',
                        type=str,
                        required=True,
                        help='path to the mesh set')
    parser.add_argument('--info',
                        type=str,
                        required=True,
                        help='path to the grasp info folder')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='path to the save the rendered data')
    parser.add_argument('--scene',
                        type=int,
                        default=0,
                        help='whether to save scene data')
    parser.add_argument('--sdf',
                        type=int,
                        default=1,
                        help='whether to save sdf data')
    parser.add_argument('--gui',
                        type=int,
                        default=0,
                        help='choose 0 for DIRECT mode and 1 (or others) for GUI mode.')
    parser.add_argument('--cuda_device',
                        default='0',
                        type=str,
                        help='id of nvidia device.')
    main(parser.parse_args())
