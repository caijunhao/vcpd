from scipy.spatial.transform.rotation import Rotation
from cpn.model import CPN
from cpn.utils import sample_contact_points, select_gripper_pose
from sim.camera import Camera
from sim.objects import RigidObject, PandaGripper
from sim.tray import Tray
from sim.utils import *
from sdf import SDF
import pybullet as p
import pybullet_data
import numpy as np
import torch
import argparse
import json
import os


def main(args):
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
    mesh_list = os.listdir(args.mesh)
    angles = np.arange(cfg['num_angle']) / cfg['num_angle'] * 2 * np.pi
    basic_rot_mats = np.expand_dims(basic_rot_mat(angles, 'y'), axis=0)  # 1*num_angle*3*3
    # tsdf initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpn = CPN()
    cpn.load_network_state_dict(device=device, pth_file=args.model_path)
    cpn.to(device)
    vol_bnd = np.array([cfg['sdf']['x_min'], cfg['sdf']['y_min'], cfg['sdf']['z_min'],
                        cfg['sdf']['x_max'], cfg['sdf']['y_max'], cfg['sdf']['z_max']]).reshape(2, 3)
    voxel_length = cfg['sdf']['resolution']
    tsdf = SDF(vol_bnd, voxel_length, rgb=False, device=device)
    i = 0
    while i < args.num_test:
        dynamic_list = list()
        static_list = list()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for j in range(cfg['test']['obj']):
            mesh_name = np.random.choice(mesh_list)
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
            body_params['baseMass'] = 0
            body_params['basePosition'] = np.array([1 - 0.5 * j, 2, 0.1])
            static_list.append(RigidObject(mesh_name,
                                           vis_params=vis_params,
                                           col_params=col_params,
                                           body_params=body_params))
            static_list[-1].change_color()
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
        init_tray.set_pose([-2, 0, 0])
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
            dynamic_list[j].set_pose(np.array([1 - 0.5 * j, 1, 0.1]), np.array([0, 0, 0, 1]))
            static_list[j].set_pose(pos, quat)
            poses[j, :3], poses[j, 3:] = pos, quat
        rgb_list, depth_list, pose_list, intr_list = list(), list(), list(), list()
        for j in range(cfg['test']['frame']):
            rgb, depth, mask = cam.get_camera_image()
            rgb_list.append(rgb), depth_list.append(depth), pose_list.append(cam.pose), intr_list.append(cam.intrinsic)
            # noise_depth = camera.add_noise(depth)
            cam.sample_a_pose_from_a_sphere(np.array(cfg['camera']['target_position']),
                                            cfg['camera']['eye_position'][-1])
        for ri, di, pi, ii, idx in zip(rgb_list, depth_list, pose_list, intr_list, range(len(intr_list))):
            ri = ri[..., 0:3].astype(np.float32)
            di = cam.add_noise(di).astype(np.float32)
            pi, ii = pi.astype(np.float32), ii.astype(np.float32)
            tsdf.integrate(di, ii, pi, rgb=ri)
            if idx in cfg['test']['test_volume']:
                cp1, cp2 = sample_contact_points(tsdf)
                ids_cp1, ids_cp2 = tsdf.get_ids(cp1), tsdf.get_ids(cp2)
                sample = dict()
                sample['sdf_volume'] = tsdf.gaussian_blur(tsdf.post_processed_volume).unsqueeze(dim=0).unsqueeze(dim=0)
                sample['ids_cp1'] = ids_cp1.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
                sample['ids_cp2'] = ids_cp2.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
                out = torch.squeeze(cpn.forward(sample))
                pos, rot, width = select_gripper_pose(tsdf, pg.vertex_sets, out, cp1, cp2, cfg['gripper']['depth'])
                quat = Rotation.from_matrix(rot).as_quat()
                pg.set_pose(pos, quat)
                pg.set_gripper_width(width + 0.02)
                print('is collided: {}'.format(pg.is_collided([])))
                pg.set_pose([-1, 0, 0], [0, 0, 0, 1])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        [p.removeBody(o.obj_id) for o in dynamic_list]
        [p.removeBody(o.obj_id) for o in static_list]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        tsdf.reset()


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
    parser.add_argument('--model_path',
                        default='',
                        type=str,
                        help='path to the pretrained model.')
    parser.add_argument('--num_test',
                        type=int,
                        default=100,
                        help='the number of test trials.')
    parser.add_argument('--gui',
                        type=int,
                        default=1,
                        help='choose 0 for DIRECT mode and 1 (or others) for GUI mode.')
    parser.add_argument('--cuda_device',
                        default='0',
                        type=str,
                        help='id of nvidia device.')
    main(parser.parse_args())
