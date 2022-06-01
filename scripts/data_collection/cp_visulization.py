from sim.utils import get_multi_body_template, add_sphere
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

    p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.57, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[0, 0, 0])
    mesh_list = os.listdir(args.mesh)
    np.random.shuffle(mesh_list)
    for mesh_name in mesh_list:
        mesh_name = '4'
        print('loading {} into pybullet...'.format(mesh_name))
        vis_params, col_params, body_params = get_multi_body_template()
        vis_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_vis.obj')
        col_params['fileName'] = os.path.join(args.mesh, mesh_name, mesh_name + '_col.obj')
        ro = RigidObject(mesh_name,
                         vis_params=vis_params,
                         col_params=col_params,
                         body_params=body_params)
        ro.change_color(rgb=[224/255, 1, 1], a=0.7)
        with open(os.path.join(args.info, mesh_name + '_info.json'), 'r') as f:
            keys = json.load(f)['keys']
        info = {k: np.load(os.path.join(args.info, mesh_name + '_' + k + '.npy')) for k in keys}
        col_flag = np.any(np.logical_not(info['collisions']), axis=1)  # col-free flag
        ref_col_flag = np.any(np.logical_not(info['collision_refine']), axis=1)  # ref-col-free flag
        centers = info['centers']# [ref_col_flag]  # [np.logical_not(ref_col_flag)]
        intersects = info['intersects']# [ref_col_flag]  # [np.logical_not(ref_col_flag)]
        vertices = 2 * centers - intersects
        vectors = vertices - intersects
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        cp1s = [add_sphere(v, radius=0.0005, rgb=[255/255, 0/255, 69/255]) for v in vertices]
        ns = [p.addUserDebugLine(vertices[i]+vectors[i]*0.005, vertices[i],
                                 lineColorRGB=np.random.uniform(size=3),
                                 lineWidth=0.15) for i in range(vertices.shape[0])]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        cp2s = [add_sphere(v, radius=0.0005, rgb=[144/255, 238/255, 144/255]) for v in intersects]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        lines = [p.addUserDebugLine(vertices[i]+vectors[i]*0.005, intersects[i]-vectors[i]*0.005,
                                    lineColorRGB=np.random.uniform(size=3),
                                    lineWidth=0.15) for i in range(vertices.shape[0])]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        [p.removeBody(cp) for cp in cp1s]
        [p.removeBody(cp) for cp in cp2s]
        [p.removeUserDebugItem(line) for line in lines]
        [p.removeUserDebugItem(line) for line in ns]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
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
