from sim.utils import basic_rot_mat, get_multi_body_template
from sim.objects import RigidObject, PandaGripper
from scipy.spatial.transform import Rotation
import argparse
import pybullet as p
import numpy as np
import trimesh
import time
import json
import os


def main(args):
    with open(args.config, 'r') as config_file:
        cfg = json.load(config_file)
    # PyBullet initialization
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.57, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0, 0, 0])
    pg = PandaGripper('assets')
    pg.set_pose([-2, -2, -2], [0, 0, 0, 1])
    mesh_list = os.listdir(args.mesh)
    np.random.shuffle(mesh_list)
    angles = np.arange(cfg['num_angle']) / cfg['num_angle'] * 2 * np.pi
    basic_rot_mats = np.expand_dims(basic_rot_mat(angles, 'y'), axis=0)
    for mesh_name in mesh_list:
        print('loading {} into pybullet...'.format(mesh_name))
        vis_params, col_params, body_params = get_multi_body_template()
        vis_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_vis.obj')
        col_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_col.obj')
        ro = RigidObject(mesh_name,
                         vis_params=vis_params,
                         col_params=col_params,
                         body_params=body_params)
        with open(os.path.join(args.info, mesh_name + '_info.json'), 'r') as f:
            keys = json.load(f)['keys']
        info = {k: np.load(os.path.join(args.info, mesh_name + '_' + k + '.npy')) for k in keys}
        col_flag = np.logical_not(info['collisions'])  # col-free flag
        ref_col_flag = np.logical_not(info['collision_refine'])  # ref-col-free flag
        bases0 = np.expand_dims(Rotation.from_quat(info['quaternions']).as_matrix(), axis=1)  # n*1*3*3
        rots0 = np.matmul(bases0, basic_rot_mats)
        centers0 = np.stack([info['centers']] * cfg['num_angle'], axis=1)
        w0 = np.stack([info['widths']] * cfg['num_angle'], axis=1)
        pos_flag = ref_col_flag
        neg_flag = np.logical_xor(col_flag, ref_col_flag)
        for flag in [pos_flag, neg_flag]:
            rots, centers, intersects, w = rots0[flag], centers0[flag], neg_flag[flag], w0[flag]
            ids = np.random.choice(np.arange(rots.shape[0]), min(20, rots.shape[0]), replace=False)
            for idx in ids:
                quat = Rotation.from_matrix(rots[idx]).as_quat()
                gripper_pos = centers[idx] - rots[idx][:, 2] * cfg['gripper']['depth']
                pg.set_pose(gripper_pos, quat)
                pg.set_gripper_width(w[idx] + 0.02)
        p.removeBody(ro.obj_id)


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
    main(parser.parse_args())
