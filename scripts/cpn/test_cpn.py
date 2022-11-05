from scipy.spatial.transform.rotation import Rotation
from cpn.model import CPN
from cpn.utils import sample_contact_points, select_gripper_pose, clustering
from sim.camera import Camera
from sim.objects import RigidObject, PandaGripper
from sim.vis import rotate_gripper
from sim.tray import Tray
from sim.utils import *
from sdf import TSDF, PSDF, GradientSDF
import xml.etree.ElementTree as et
import pymeshlab as ml
import pybullet as p
import pybullet_data
import numpy as np
import torch
import argparse
import json
import time
import os


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    with open(args.config, 'r') as config_file:
        cfg = json.load(config_file)
    # PyBullet initialization
    mode = p.GUI if args.gui else p.DIRECT
    p.connect(mode, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.43, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0, 0, 0])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF('plane.urdf')
    # scene initialization
    tray = Tray(**cfg['tray'])
    tray.set_pose([-10, -10, -10])
    cfg['tray']['height'], cfg['tray']['width'], cfg['tray']['theta'] = 0.324, 0.324, 80
    init_tray = Tray(**cfg['tray'])
    cam = Camera(**cfg['camera'])
    pg = PandaGripper('assets')
    ms = ml.MeshSet()
    vertex_sets = dict()
    for component in pg.components:
        col2_path = os.path.join('assets', component + '_col2.obj')
        ms.load_new_mesh(col2_path)
        vertex_sets[component] = ms.current_mesh().vertex_matrix()
    pg.set_pose([-20, -20, -20], [0, 0, 0, 1])
    mesh_list = os.listdir(args.mesh)
    # sdf initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpn = CPN()
    cpn.load_network_state_dict(device=device, pth_file=args.model_path)
    cpn.to(device)
    voxel_length = cfg['sdf']['voxel_length']
    vol_bnd = np.array([cfg['sdf']['x_min'], cfg['sdf']['y_min'], cfg['sdf']['z_min'],
                        cfg['sdf']['x_max'], cfg['sdf']['y_max'], cfg['sdf']['z_max']]).reshape(2, 3)
    origin = vol_bnd[0]
    resolution = np.ceil((vol_bnd[1] - vol_bnd[0] - voxel_length / 7) / voxel_length).astype(np.int)
    sdf = TSDF(origin, resolution, voxel_length, fuse_color=False, device=device)
    i = 0
    avg_anti_score = 0.0
    col_free_rate = 0.0
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
            pos, quat = np.array([1 - 0.5 * j, -10, -0.1]), np.array([0, 0, 0, 1])
            body_params['basePosition'], body_params['baseOrientation'] = pos, quat
            body_params['baseMass'] = 1.0
            # dynamic_list.append(RigidObject(mesh_name,
            #                                 fileName=urdf_path, basePosition=pos, baseOrientation=quat, mass=1.0))
            dynamic_list.append(RigidObject(mesh_name,
                                            vis_params=vis_params,
                                            col_params=col_params,
                                            body_params=body_params))
            body_params['baseMass'] = 0
            body_params['basePosition'] = np.array([1 - 0.5 * j, -11, -0.1])
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
        init_tray.set_pose([-20, -11, -1])
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
            dynamic_list[j].set_pose(np.array([1 - 0.5 * j, -10, -0.1]), np.array([0, 0, 0, 1]))
            static_list[j].set_pose(pos, quat)
            poses[j, :3], poses[j, 3:] = pos, quat
        rgb_list, depth_list, pose_list, intr_list = list(), list(), list(), list()
        for j in range(cfg['test']['frame']):
            rgb, depth, mask = cam.get_camera_image()
            rgb_list.append(rgb), depth_list.append(depth), pose_list.append(cam.pose), intr_list.append(cam.intrinsic)
            if cfg['test']['pose_sampling'] == 'sphere':
                radius = np.random.uniform(cfg['camera']['z_min'], cfg['camera']['z_max'])
                cam.sample_a_pose_from_a_sphere(np.array(cfg['camera']['target_position']), radius)
            elif cfg['test']['pose_sampling'] == 'cube':
                cam.sample_a_position(cfg['camera']['x_min'], cfg['camera']['x_max'],
                                      cfg['camera']['y_min'], cfg['camera']['y_max'],
                                      cfg['camera']['z_min'], cfg['camera']['z_max'],
                                      cfg['camera']['up_vector'])
        for ri, di, pi, ii, idx in zip(rgb_list, depth_list, pose_list, intr_list, range(len(intr_list))):
            ri = ri[..., 0:3].astype(np.float32)
            di = cam.add_noise(di).astype(np.float32)
            pi, ii = pi.astype(np.float32), ii.astype(np.float32)
            sdf.integrate(di, ii, pi, rgb=ri)
        cp1, cp2 = sample_contact_points(sdf)
        ids_cp1, ids_cp2 = sdf.get_ids(cp1), sdf.get_ids(cp2)
        sample = dict()
        sample['sdf_volume'] = sdf.gaussian_smooth(sdf.post_processed_vol).unsqueeze(dim=0).unsqueeze(dim=0)
        sample['ids_cp1'] = ids_cp1.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
        sample['ids_cp2'] = ids_cp2.permute((1, 0)).unsqueeze(dim=0).unsqueeze(dim=-1)
        out = torch.squeeze(cpn.forward(sample))
        gripper_pos, rot, width, cp1, cp2 = select_gripper_pose(sdf, vertex_sets, out, cp1, cp2,
                                                                cfg['gripper']['depth'], check_tray=False)
        # debug: uncomment for visualization
        # visualize_contacts(cp1.cpu().numpy(), cp2.cpu().numpy(), num_vis=777)
        # /debug
        gripper_pos, rot, width, cp1, cp2 = clustering(gripper_pos, rot, width, cp1, cp2)
        end_point_pos = gripper_pos + rot[:, 2] * cfg['gripper']['depth']
        closest_obj = get_closest_obj(end_point_pos, static_list)
        contact1, normal1, contact2, normal2 = get_contact_points(cp1, cp2, closest_obj)
        if contact1 is not None:
            grasp_direction = contact2 - contact1
            grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
            curr_anti_score = np.abs(grasp_direction @ normal1) * np.abs(grasp_direction @ normal2)
        else:
            curr_anti_score = 0
        quat = Rotation.from_matrix(rot).as_quat()
        # rotate_gripper(end_point_pos, rot, width, cfg['gripper']['depth'], tray.get_tray_ids())
        pg.set_pose(gripper_pos, quat)
        pg.set_gripper_width(width)
        # debug: uncomment for visualization
        # visualize_contact(contact1, normal1, contact2, normal2)
        # closest_obj.change_color()
        # /debug
        # sdf.write_mesh('out.ply', *sdf.compute_mesh(step_size=1))
        print('current antipodal score: {:04f} given {} view(s)'.format(curr_anti_score, len(intr_list)))
        col = pg.is_collided(tray.get_tray_ids())  # , show_col=True
        print('is collided: {}'.format(col))
        pg.set_pose([-10, -10, -1], [0, 0, 0, 1])
        avg_anti_score = (curr_anti_score + avg_anti_score * i) / (i + 1)
        print('# of trials: {} | current average antipodal score: {:04f}'.format(i, avg_anti_score))
        col_free_rate = (1 - int(col) + col_free_rate * i) / (i + 1)
        print('# of trials: {} | current average collision free rate: {:04f}'.format(i, col_free_rate))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        [p.removeBody(o.obj_id) for o in dynamic_list]
        [p.removeBody(o.obj_id) for o in static_list]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        sdf.reset()
        cam.randomize_fov()
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
    parser.add_argument('--model_path',
                        default='',
                        type=str,
                        help='path to the pretrained model.')
    parser.add_argument('--num_test',
                        type=int,
                        default=1000,
                        help='the number of test trials.')
    parser.add_argument('--gui',
                        type=int,
                        default=1,
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
