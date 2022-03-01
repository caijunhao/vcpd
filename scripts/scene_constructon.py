from sim.utils import *
from sim.objects import PandaGripper, RigidObject
from sim.camera import Camera
from sim.tray import Tray
from sdf import TSDF
from scipy.spatial.transform import Rotation
from skimage import io
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import numpy as np
import torch
import trimesh
import argparse
import json
import time
import os

torch.set_default_dtype(torch.float32)
tf32 = torch.float32
nf32 = np.float32


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
    # tsdf initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol_bnd = np.array([cfg['sdf']['x_min'], cfg['sdf']['x_max'],
                        cfg['sdf']['y_min'], cfg['sdf']['y_max'],
                        cfg['sdf']['z_min'], cfg['sdf']['z_max']], dtype=nf32).reshape(3, 2)
    res = cfg['sdf']['resolution']
    tsdf = TSDF(vol_bnd, res, rgb=False, device=device)
    vol_bnd = tsdf.vol_bnd
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
            # retrieve negative contact points
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
        flag = check_valid_pts(contact_pts1, vol_bnd.T) * check_valid_pts(contact_pts2, vol_bnd.T)
        contact_pts1, contact_pts2 = contact_pts1[flag], contact_pts2[flag]
        contact_ids = np.arange(contact_pts1.shape[0])
        np.random.shuffle(contact_ids)  # randomly permute the order of right contact points to create neg samples
        neg1_pts1 = contact_pts1.copy()
        neg1_pts2 = contact_pts2.copy()[contact_ids]
        neg2_pts1 = np.concatenate(neg_pts1, axis=0).astype(np.float32)
        neg2_pts2 = np.concatenate(neg_pts2, axis=0).astype(np.float32)
        neg_pts1 = np.concatenate([neg1_pts1, neg2_pts1], axis=0)
        neg_pts2 = np.concatenate([neg1_pts2, neg2_pts2], axis=0)
        flag = check_valid_pts(neg_pts1, vol_bnd.T) * check_valid_pts(neg_pts2, vol_bnd.T)
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
                cam.sample_a_pose_from_a_sphere(np.array(cfg['camera']['target_position']),
                                                cfg['camera']['eye_position'][-1])
            elif cfg['scene']['pose_sampling'] == 'cube':
                cam.sample_a_position(cfg['camera']['x_min'], cfg['camera']['x_max'],
                                      cfg['camera']['y_min'], cfg['camera']['y_max'],
                                      cfg['camera']['z_min'], cfg['camera']['z_max'],
                                      cfg['camera']['up_vector'])
            else:
                raise ValueError('only sphere or cube sampling methods are supported')
        e = time.time()
        print('elapse time on scene rendering: {}s'.format(e - b))
        # tsdf generation
        sdf_path = os.path.join(args.output, '{:06d}'.format(i))
        if args.sdf:
            for ri, di, pi, ii, idx in zip(rgb_list, depth_list, pose_list, intr_list, range(len(intr_list))):
                os.makedirs(sdf_path) if not os.path.exists(sdf_path) else None
                ri = ri[..., 0:3].astype(np.float32)
                di = cam.add_noise(di).astype(np.float32)
                pi, ii = pi.astype(np.float32), ii.astype(np.float32)
                tsdf.tsdf_integrate(di, ii, pi, rgb=ri)
                if idx in cfg['sdf']['save_volume']:
                    sdf_cp1, sdf_cp2 = tsdf.extract_sdf(contact_pts1), tsdf.extract_sdf(contact_pts2)
                    sdf_cp_flag = np.logical_and(np.abs(sdf_cp1) <= 0.2, np.abs(sdf_cp2) <= 0.2)
                    val_pts1, val_pts2 = contact_pts1[sdf_cp_flag], contact_pts2[sdf_cp_flag]
                    sdf_ncp1, sdf_ncp2 = tsdf.extract_sdf(neg_pts1), tsdf.extract_sdf(neg_pts2)
                    sdf_ncp_flag = np.logical_and(np.abs(sdf_ncp1) <= 0.2, np.abs(sdf_ncp2) <= 0.2)
                    val_n_pts1, val_n_pts2 = neg_pts1[sdf_ncp_flag], neg_pts2[sdf_ncp_flag]
                    num_cp = val_pts1.shape[0]
                    num_ncp = val_n_pts1.shape[0]
                    if num_cp == 0 or num_ncp == 0:
                        print('no valid point was found, discard current scene')
                        continue
                    selected_ids = np.random.choice(np.arange(num_ncp), num_cp, replace=num_ncp < num_cp)
                    print('num_cp: {} | num_ncp: {}'.format(num_cp, min(num_cp, num_ncp)))
                    val_n_pts1, val_n_pts2 = val_n_pts1[selected_ids], val_n_pts2[selected_ids]
                    if cfg['sdf']['gaussian_blur']:
                        sdf_vol = tsdf.gaussian_blur(tsdf.post_processed_volume)
                    else:
                        sdf_vol = tsdf.post_processed_volume
                    sdf_vol_cpu = sdf_vol.cpu().numpy()
                    np.save(os.path.join(sdf_path, '{:04d}_pos_contact1.npy'.format(idx)), tsdf.get_ids(val_pts1))
                    np.save(os.path.join(sdf_path, '{:04d}_pos_contact2.npy'.format(idx)), tsdf.get_ids(val_pts2))
                    np.save(os.path.join(sdf_path, '{:04d}_neg_contact1.npy'.format(idx)), tsdf.get_ids(val_n_pts1))
                    np.save(os.path.join(sdf_path, '{:04d}_neg_contact2.npy'.format(idx)), tsdf.get_ids(val_n_pts2))
                    np.save(os.path.join(sdf_path, '{:04d}_sdf_volume.npy'.format(res, idx)), sdf_vol_cpu)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        [p.removeBody(o.obj_id) for o in dynamic_list]
        [p.removeBody(o.obj_id) for o in static_list]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        tsdf.write_mesh(os.path.join(sdf_path, '{:06d}_mesh.ply'.format(i)),
                        *tsdf.compute_mesh(step_size=3))
        tsdf.reset()
        i += 1


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
                        default='1',
                        type=str,
                        help='id of nvidia device.')
    main(parser.parse_args())
