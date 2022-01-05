from sim.utils import get_multi_body_template, sample_a_pose, step_simulation, basic_rot_mat
from sim.objects import PandaGripper, RigidObject
from sim.camera import Camera
from sim.tray import Tray
from scipy.spatial.transform import Rotation
from skimage import io
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import numpy as np
import trimesh
import argparse
import json
import time
import os


def main(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    mode = p.GUI if args.gui else p.DIRECT
    p.connect(mode)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF('plane.urdf')
    with open(args.config, 'r') as config_file:
        cfg = json.load(config_file)
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
    basic_rot_mats = np.expand_dims(basic_rot_mat(angles, 'x'), axis=0)  # 1*num_angle*3*3
    for i in range(cfg['scene']['trial']):
        info_dict = dict()
        dynamic_list = list()
        static_list = list()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for j in range(cfg['scene']['obj']):
            mesh_name = np.random.choice(mesh_list)
            if mesh_name not in info_dict:
                with open(os.path.join(args.info, mesh_name+'_info.json'), 'r') as f:
                    keys = json.load(f)['keys']
                info_dict[mesh_name] = {k: np.load(os.path.join(args.info, mesh_name+'_'+k+'.npy')) for k in keys}
            print('loading {} into pybullet...'.format(mesh_name))
            vis_params, col_params, body_params = get_multi_body_template()
            vis_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name+'_vis.obj')
            col_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name+'_col.obj')
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
        name_list = np.array([o.obj_name for o in dynamic_list])
        poses = np.zeros((cfg['scene']['obj'], 7))
        col_free_list = list()
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
            # col_flag = np.sum(cols, axis=1).astype(bool)
            col_free = col_flag[col_flag]
            bases0 = np.expand_dims(Rotation.from_quat(info_dict[o.obj_name]['quaternions']).as_matrix(),
                                    axis=1)  # n*1*3*3
            rots0 = np.matmul(bases0, basic_rot_mats)[col_flag]
            centers0 = np.stack([info_dict[o.obj_name]['centers']]*cfg['num_angle'], axis=1)[col_flag]
            w0 = np.stack([info_dict[o.obj_name]['widths']]*cfg['num_angle'], axis=1)[col_flag]
            rots = np.matmul(pose[0:3, 0:3].reshape(1, 3, 3), rots0)
            quats = Rotation.from_matrix(rots).as_quat()
            centers = centers0 @ pose[0:3, 0:3].T + pose[0:3, 3].reshape(1, 3)
            z_axes = rots[..., :, 2]
            z_flag = z_axes[..., 2] < 0
            for k in np.arange(z_flag.shape[0])[z_flag]:
                pos = centers[k] - z_axes[k] * cfg['gripper']['depth']
                pg.set_pose(pos, quats[k])
                pg.set_gripper_width(w0[k]+0.02)
                col_free[k] = not pg.is_collided(exemption)
            col_free_list.append(col_free)
        pg.set_pose([-2, -2, -2], [0, 0, 0, 1])
        e = time.time()
        print('elapse time on collision checking: {}s'.format(e-b))
        cam.set_pose(cfg['camera']['eye_position'], cfg['camera']['target_position'], cfg['camera']['up_vector'])
        scene_path = os.path.join(args.output, '{:06d}'.format(i))
        os.makedirs(scene_path, exist_ok=True)
        print('render scene images...')
        b = time.time()
        for j in range(cfg['scene']['frame']):
            rgb, depth, mask = cam.get_camera_image()
            # noise_depth = camera.add_noise(depth)
            io.imsave(os.path.join(scene_path, '{:04d}_rgb.png'.format(j)), rgb, check_contrast=False)
            io.imsave(os.path.join(scene_path, '{:04d}_encoded_depth.png'.format(j)), cam.encode_depth(depth), check_contrast=False)
            # plt.imsave(os.path.join(scene_path, '{:04d}_depth.png'.format(j)), depth)
            # io.imsave(os.path.join(scene_path, '{:04d}_mask.png'.format(k)), mask, check_contrast=False)
            np.save(os.path.join(scene_path, '{:04d}_pos&quat.npy'.format(j)), np.concatenate(cam.get_pose()))
            np.save(os.path.join(scene_path, '{:04d}_intrinsic.npy'.format(j)), cam.intrinsic)
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
        np.save(os.path.join(scene_path, 'mesh_names.npy'), np.array(name_list))
        np.save(os.path.join(scene_path, 'col_free_flags.npy'), np.array(col_free_list, dtype=object))
        np.save(os.path.join(scene_path, 'obj_poses.npy'), poses)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        [p.removeBody(o.obj_id) for o in dynamic_list]
        [p.removeBody(o.obj_id) for o in static_list]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


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
    parser.add_argument('--gui',
                        type=int,
                        default=1,
                        help='choose 0 for DIRECT mode and 1 (or others) for GUI mode.')
    main(parser.parse_args())
