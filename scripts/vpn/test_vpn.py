from scipy.spatial.transform.rotation import Rotation
from vpn.model import VPN, RefineNetV0
from vpn.utils import rank_and_group_poses, DiscretizedGripper
from sim.camera import Camera
from sim.objects import RigidObject, PandaGripper
from sim.tray import Tray
from sim.utils import *
from sdf import SDF
import xml.etree.ElementTree as et
import pybullet as p
import pybullet_data
import numpy as np
import trimesh
import torch
import argparse
import json
import os


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
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
    dg = DiscretizedGripper(cfg['refine'])
    mesh_list = os.listdir(args.mesh)
    angles = np.arange(cfg['num_angle']) / cfg['num_angle'] * 2 * np.pi
    # tsdf initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vpn = VPN()
    rn = RefineNetV0(num_sample=cfg['refine']['num_sample'])
    vpn.load_network_state_dict(device=device, pth_file=args.vpn_path)
    rn.load_network_state_dict(device=device, pth_file=args.rn_path)
    vpn.to(device)
    rn.to(device)
    vol_bnd = np.array([cfg['sdf']['x_min'], cfg['sdf']['y_min'], cfg['sdf']['z_min'],
                        cfg['sdf']['x_max'], cfg['sdf']['y_max'], cfg['sdf']['z_max']]).reshape(2, 3)
    voxel_length = cfg['sdf']['resolution']
    tsdf = SDF(vol_bnd, voxel_length, rgb=False, device=device)
    panda_gripper_mesh = trimesh.load_mesh('assets/panda_gripper_col4.ply')
    gpr_pts = torch.from_numpy(panda_gripper_mesh.vertices.astype(np.float32)).to(device)
    i = 0
    avg_anti_score_vpn, avg_anti_score_rn = 0.0, 0.0
    col_free_rate_vpn, col_free_rate_rn = 0.0, 0.0
    while i < args.num_test:
        dynamic_list = list()
        static_list = list()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for j in range(cfg['test']['obj']):
            mesh_name = np.random.choice(mesh_list)
            print('loading {} into pybullet...'.format(mesh_name))
            urdf_path = os.path.join(args.mesh, mesh_name, mesh_name + '.urdf')
            scale = [elem for elem in et.parse(urdf_path).getroot().iter('mesh')][0].attrib['scale'].split(' ')
            scale = [float(s) for s in scale]
            vis_params, col_params, body_params = get_multi_body_template()
            vis_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_vis.obj')
            vis_params['meshScale'] = scale
            col_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_col.obj')
            col_params['meshScale'] = scale
            pos, quat = np.array([1 - 0.5 * j, 1, -0.1]), np.array([0, 0, 0, 1])
            body_params['basePosition'], body_params['baseOrientation'] = pos, quat
            body_params['baseMass'] = 1.0
            # dynamic_list.append(RigidObject(mesh_name,
            #                                 fileName=urdf_path, basePosition=pos, baseOrientation=quat, mass=1.0))
            dynamic_list.append(RigidObject(mesh_name,
                                            vis_params=vis_params,
                                            col_params=col_params,
                                            body_params=body_params))
            body_params['baseMass'] = 0
            body_params['basePosition'] = np.array([1 - 0.5 * j, 2, -0.1])
            static_list.append(RigidObject(mesh_name,
                                           vis_params=vis_params,
                                           col_params=col_params,
                                           body_params=body_params))
            # static_list.append(RigidObject(mesh_name,
            #                                fileName=urdf_path,
            #                                basePosition=np.array([1 - 0.5 * j, 2, 0.1]),
            #                                baseOrientation=np.array([0, 0, 0, 1]),
            #                                mass=0))
            static_list[-1].change_color(a=0.7)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        place_order = np.arange(cfg['test']['obj'])
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
        init_tray.set_pose([-2, 0, -1])
        tray.set_pose([0, 0, 0])
        stable = False
        curr_wait = 0
        while not stable and curr_wait < 3:
            stable = dynamic_list[0].is_stable() & dynamic_list[-1].is_stable()
            curr_wait += 1
        print('scene stable, start generating data.')
        poses = np.zeros((cfg['test']['obj'], 7))
        # replace dynamic objects with static ones to keep the scene stable during rendering
        for j in place_order:
            pos, quat = dynamic_list[j].get_pose()
            dynamic_list[j].set_pose(np.array([1 - 0.5 * j, 1, -0.1]), np.array([0, 0, 0, 1]))
            static_list[j].set_pose(pos, quat)
            poses[j, :3], poses[j, 3:] = pos, quat
        rgb_list, depth_list, pose_list, intr_list = list(), list(), list(), list()
        for j in range(cfg['test']['frame']):
            rgb, depth, mask = cam.get_camera_image()
            rgb_list.append(rgb), depth_list.append(depth), pose_list.append(cam.pose), intr_list.append(cam.intrinsic)
            # noise_depth = camera.add_noise(depth)
            cam.sample_a_position(cfg['camera']['x_min'], cfg['camera']['x_max'],
                                  cfg['camera']['y_min'], cfg['camera']['y_max'],
                                  cfg['camera']['z_min'], cfg['camera']['z_max'],
                                  cfg['camera']['up_vector'])
        curr_anti_score, col = 0.0, 1
        for ri, di, pi, ii, idx in zip(rgb_list, depth_list, pose_list, intr_list, range(len(intr_list))):
            ri = ri[..., 0:3].astype(np.float32)
            di = cam.add_noise(di).astype(np.float32)
            pi, ii = pi.astype(np.float32), ii.astype(np.float32)
            tsdf.integrate(di, ii, pi, rgb=ri)
        sample = dict()
        v, _, n, _ = tsdf.compute_mesh()
        v, n = torch.from_numpy(v).to(device), torch.from_numpy(n).to(device)
        ids = tsdf.get_ids(v)
        sample['sdf_volume'] = tsdf.gaussian_blur(tsdf.post_processed_volume).unsqueeze(dim=0).unsqueeze(dim=0)
        sample['pts'] = v.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
        sample['normals'] = n.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
        sample['ids'] = ids.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
        sample['occupancy_volume'] = torch.zeros_like(sample['sdf_volume'], device=device)
        sample['occupancy_volume'][sample['sdf_volume'] <= 0] = 1
        sample['origin'] = tsdf.origin
        sample['resolution'] = torch.tensor([[tsdf.res]], dtype=torch.float32, device=device)
        out = torch.sigmoid(vpn.forward(sample))
        groups = rank_and_group_poses(sample, out, device, gpr_pts, collision_check=True)
        if groups is None:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            [p.removeBody(o.obj_id) for o in dynamic_list]
            [p.removeBody(o.obj_id) for o in static_list]
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            pg.set_pose([-1, 0, -1], [0, 0, 0, 1])
            tsdf.reset()
            continue
        score, pose = groups[0]['queue'].get()
        pos, rot0 = pose[0:3], Rotation.from_quat(pose[6:10]).as_matrix()
        rot = rot0 @ basic_rot_mat(np.pi / 2, axis='z').astype(np.float32)
        quat = Rotation.from_matrix(rot).as_quat()
        gripper_pos = pos - 0.08 * rot[:, 2]
        end_point_pos = gripper_pos + rot[:, 2] * cfg['gripper']['depth']
        closest_obj = get_closest_obj(end_point_pos, static_list)
        contact1, normal1, contact2, normal2 = get_contact_points_from_center(end_point_pos,
                                                                              rot[:, 1],
                                                                              closest_obj)
        if contact1 is not None:
            grasp_direction = contact2 - contact1
            grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
            curr_anti_score = np.abs(grasp_direction @ normal1) * np.abs(grasp_direction @ normal2)
            # uncomment for visualization
            # l0 = p.addUserDebugLine(contact1, contact2)
            # s1 = add_sphere(contact1)
            # l1 = p.addUserDebugLine(contact1 - 0.01 * normal1, contact1 + 0.01 * normal1)
            # s2 = add_sphere(contact2)
            # l2 = p.addUserDebugLine(contact2 - 0.01 * normal2, contact2 + 0.01 * normal2)
            # p.removeBody(s1), p.removeBody(s2)
            # p.removeUserDebugItem(l0), p.removeUserDebugItem(l1), p.removeUserDebugItem(l2)
            # closest_obj.change_color()
            # tsdf.write_mesh('out.ply', *tsdf.compute_mesh(step_size=1))
        else:
            curr_anti_score = 0
        pg.set_pose(gripper_pos, quat)
        print('current antipodal score: {:04f} given {} view(s)'.format(curr_anti_score, len(intr_list)))
        col = pg.is_collided(tray.get_tray_ids())
        print('is collided: {}'.format(col))
        # pg.set_pose([-1, 0, -1], [0, 0, 0, 1])
        avg_anti_score_vpn = (curr_anti_score + avg_anti_score_vpn * i) / (i + 1)
        print('# of trials: {} | current average antipodal score for vpn: {:04f}'.format(i, avg_anti_score_vpn))
        col_free_rate_vpn = (1 - int(col) + col_free_rate_vpn * i) / (i + 1)
        print('# of trials: {} | current average collision free rate for vpn: {:04f}'.format(i, col_free_rate_vpn))
        # grasp pose refinement network
        for _ in range(5):
            trans = (pose[0:3] + pose[3:6] * 0.01).astype(np.float32)
            rot = Rotation.from_quat(pose[6:10]).as_matrix().astype(np.float32)
            pos = dg.sample_grippers(trans.reshape(1, 3), rot.reshape(1, 3, 3), inner_only=False)
            pos = np.expand_dims(pos.transpose((2, 1, 0)), axis=0)  # 1 * 3 * 2048 * 1
            sample['perturbed_pos'] = torch.from_numpy(pos).to(device)
            delta_rot = rn(sample)
            delta_rot = torch.squeeze(delta_rot).detach().cpu().numpy()
            rot_recover = rot @ delta_rot
            approach_recover = rot_recover[:, 2]
            quat_recover = Rotation.from_matrix(rot_recover).as_quat()
            trans_recover = trans
            # trans_recover[2] -= 0.005
            pos_recover = pose[0:3]
            pose = np.concatenate([pos_recover, -approach_recover, quat_recover])
        pos, rot0 = pose[0:3], Rotation.from_quat(pose[6:10]).as_matrix()
        rot = rot0 @ basic_rot_mat(np.pi / 2, axis='z').astype(np.float32)
        gripper_pos = pos - 0.08 * rot[:, 2]
        end_point_pos = gripper_pos + rot[:, 2] * cfg['gripper']['depth']
        closest_obj = get_closest_obj(end_point_pos, static_list)
        quat = Rotation.from_matrix(rot).as_quat()
        contact1, normal1, contact2, normal2 = get_contact_points_from_center(end_point_pos,
                                                                              rot[:, 1],
                                                                              closest_obj)
        if contact1 is not None:
            grasp_direction = contact2 - contact1
            grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
            curr_anti_score = np.abs(grasp_direction @ normal1) * np.abs(grasp_direction @ normal2)
            # uncomment for visualization
            # l0 = p.addUserDebugLine(contact1, contact2)
            # s1 = add_sphere(contact1)
            # l1 = p.addUserDebugLine(contact1 - 0.01 * normal1, contact1 + 0.01 * normal1)
            # s2 = add_sphere(contact2)
            # l2 = p.addUserDebugLine(contact2 - 0.01 * normal2, contact2 + 0.01 * normal2)
            # p.removeBody(s1), p.removeBody(s2)
            # p.removeUserDebugItem(l0), p.removeUserDebugItem(l1), p.removeUserDebugItem(l2)
            # closest_obj.change_color()
            # tsdf.write_mesh('out.ply', *tsdf.compute_mesh(step_size=1))
        else:
            curr_anti_score = 0
        pg.set_pose(gripper_pos, quat)
        print('current antipodal score: {:04f} given {} view(s)'.format(curr_anti_score, len(intr_list)))
        col = pg.is_collided(tray.get_tray_ids())
        print('is collided: {}'.format(col))
        pg.set_pose([-1, 0, -1], [0, 0, 0, 1])
        avg_anti_score_rn = (curr_anti_score + avg_anti_score_rn * i) / (i + 1)
        print('# of trials: {} | current average antipodal score for rn: {:04f}'.format(i, avg_anti_score_rn))
        col_free_rate_rn = (1 - int(col) + col_free_rate_rn * i) / (i + 1)
        print('# of trials: {} | current average collision free rate for rn: {:04f}'.format(i, col_free_rate_rn))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        [p.removeBody(o.obj_id) for o in dynamic_list]
        [p.removeBody(o.obj_id) for o in static_list]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
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
    parser.add_argument('--vpn_path',
                        default='',
                        type=str,
                        help='path to trained vpn model.')
    parser.add_argument('--rn_path',
                        default='',
                        type=str,
                        help='path to trained refine net model.')
    parser.add_argument('--num_test',
                        type=int,
                        default=1000,
                        help='the number of test trials.')
    parser.add_argument('--gui',
                        type=int,
                        default=0,
                        help='choose 0 for DIRECT mode and 1 (or others) for GUI mode.')
    parser.add_argument('--cuda_device',
                        default='0',
                        type=str,
                        help='id of nvidia device.')
    parser.add_argument('--seed',
                        default=177,
                        type=int,
                        help='random seed for torch and numpy random number generator.')
    main(parser.parse_args())
